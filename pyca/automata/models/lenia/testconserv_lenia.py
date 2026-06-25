import torch
import pygame

from .macelenia import MaCELenia
from .lenia import Lenia
from pyca.interface import LabeledSlider, DropDown


class TestConservLenia(MaCELenia):
    """
    Experimental MaCELenia variant for testing alternative *mass-conserving*
    discretizations of the advection equation  drho/dt = div( rho grad(A) ).

    Affinity A is computed exactly like in MaCELenia (Lenia growth field). Only
    the *redistribution* step (how mass is portioned to the 9 Moore neighbors
    according to affinity differences) changes. The method is selectable live,
    from the GUI dropdown, WITHOUT resetting the automaton state, so different
    schemes can be compared on the same configuration.

    All schemes share the same bookkeeping (column-stochastic weights summing
    to 1 over each source's neighborhood -> exact mass conservation), and are
    implemented with `torch.roll` (circular boundary, matching MaCELenia).

    Notation (per source cell s, neighbor n in the 3x3 Moore neighborhood N_s):
        Delta_{sn} = A_n - A_s        (Delta_{ss} = 0)
        rho_n^{t+1} = sum_{s : n in N_s} w_{s->n} rho_s^t,   sum_n w_{s->n} = 1.

    Methods (see the MaCE menu):

      Family A -- pull-normalized (score / neighborhood sum):
        'MaCE'        : softmax, w ∝ exp(beta * Delta)              (the paper)
        'MaCE-iso'    : softmax with 4:1 axial:diagonal isotropy stencil
        'sparsemax'   : entmax alpha=2, exact-zero portions (sharp fronts)
        'entmax'      : Tsallis alpha-entmax, alpha tunable (porous-medium)
        'winner'      : winner-take-all (beta -> inf), zero diffusion

      Family B -- push-rate with self-remainder (rate lambda is the CFL knob):
        'uphill'      : w = lambda * relu(Delta)      (no floor, pure advection)
                        (algebraically the Family C upwind donor-cell flux)
        'eps-uphill'  : w = lambda * (eps + relu(Delta))  (tunable baseline diff.)
        'metropolis'  : w = lambda/8 * min(1, exp(2*beta*Delta))   (e^{2bA} eq.)
        'barker'      : w = lambda/8 * sigmoid(2*beta*Delta)
        'exclusion'   : barker throttled by (1 - rho_n/rho_max)  (capacity limit)

      Family D -- iterative:
        'sinkhorn'    : doubly-stochastic balancing (anti-pileup), ~4 sweeps

      Semi-Lagrangian -- forward-push (cloud-in-cell), NOT a 9-stencil weight:
        'semi-lagrangian' : move each cell's parcel forward by v*dt,
                            v = 2*beta*grad(A), and scatter-add its mass onto the
                            4 surrounding cells with bilinear weights. Targets the
                            pure continuity equation d rho/dt = -div(rho v), i.e.
                            MaCE's advective limit WITHOUT the forced beta=0
                            Laplacian floor. Exactly conservative, positivity- and
                            vacuum-safe (no division by rho). Step size is the
                            'SL dt' slider; the 'diffusion' slider adds the
                            optional 5-point-Laplacian sub-step (kappa = d/4).

    The Family-B off-diagonal weights are passed through a per-cell limiter
    (scale by min(1, 1/sum_off)) so the self-remainder stays >= 0 for ANY
    lambda: this guarantees validity/positivity, the per-cell adaptive version
    of the CFL cap.

    Universal diffusion knob (`diffusion` slider, d in [0,1]): after a scheme
    produces its weights, they are convex-blended toward the uniform box-average
        w <- (1 - d) * w + d * (1/9).
    Since a convex combination of column-stochastic weights is column-stochastic,
    mass is still conserved exactly; d * box-average is a fractional heat-kernel
    step, so this adds *tunable* isotropic diffusion to schemes that otherwise
    have little/none (uphill, sparsemax, winner, ...). At d=0 every scheme is
    unchanged; MaCE already diffuses inherently, so there d only stacks on top.
    """

    # Order of the dropdown menu
    METHODS = [
        "MaCE",
        "MaCE-iso",
        "sparsemax",
        "entmax",
        "winner",
        "uphill",
        "eps-uphill",
        "metropolis",
        "barker",
        "exclusion",
        "sinkhorn",
        "semi-lagrangian",
    ]

    def __init__(
        self,
        size,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        has_food=False,
        sense_food=False,
        interest_files=None,
        save_dir=".",
    ):
        super().__init__(
            size,
            num_channels=num_channels,
            params=params,
            state_init=state_init,
            device=device,
            has_food=has_food,
            sense_food=sense_food,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        # --- Redistribution scheme + its parameters ---
        self.redist_method = "MaCE"
        self.lam = 0.3        # rate lambda for Family B (CFL knob)
        self.alpha = 1.5      # alpha for entmax (1 -> softmax, 2 -> sparsemax)
        self.eps = 0.05       # baseline diffusion floor for eps-uphill
        self.rho_max = 1.0    # capacity for exclusion
        self.diffusion = 0.0  # universal isotropic diffusion knob d in [0,1]
        self.sl_dt = 0.05     # step size dt for the semi-lagrangian push (disp = v*dt)

        # --- 3x3 Moore neighborhood offsets (row-major), center is index 4 ---
        self._offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        self._center_k = self._offsets.index((0, 0))

        # Isotropy stencil c_{sn}: axial & center = 4, diagonal = 1 (4:1 ratio)
        c_vals = []
        for (di, dj) in self._offsets:
            if di != 0 and dj != 0:
                c_vals.append(1.0)  # diagonal
            else:
                c_vals.append(4.0)  # axial or center
        self._iso_c = torch.tensor(c_vals, device=self.device).view(1, 1, 9, 1, 1)

        # --- GUI components ---
        self.method_dropdown = DropDown(
            options=self.METHODS, manager=self.manager, starting_option=self.redist_method
        )
        self.register_component(self.method_dropdown)

        self.lambda_slider = LabeledSlider(
            min_value=0.0, max_value=1.0, manager=self.manager, title="lambda (rate)",
            initial_value=self.lam, tick_size=0.01,
        )
        self.register_component(self.lambda_slider)

        self.alpha_slider = LabeledSlider(
            min_value=1.0, max_value=3.0, manager=self.manager, title="alpha (entmax)",
            initial_value=self.alpha, tick_size=0.05,
        )
        self.register_component(self.alpha_slider)

        self.eps_slider = LabeledSlider(
            min_value=0.0, max_value=0.5, manager=self.manager, title="eps (floor)",
            initial_value=self.eps, tick_size=0.005,
        )
        self.register_component(self.eps_slider)

        self.diffusion_slider = LabeledSlider(
            min_value=0.0, max_value=1.0, manager=self.manager, title="diffusion",
            initial_value=self.diffusion, tick_size=0.01,
        )
        self.register_component(self.diffusion_slider)

        self.sl_dt_slider = LabeledSlider(
            min_value=0.0, max_value=0.25, manager=self.manager, title="SL dt",
            initial_value=self.sl_dt, tick_size=0.005,
        )
        self.register_component(self.sl_dt_slider)

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #
    def step(self, sense_food=None):
        """
        Steps the model: compute affinity (like MaCELenia), then redistribute
        mass using the currently selected scheme, then the optional food step.
        """
        sf = self.sense_food if sense_food is None else sense_food
        Aff = self._compute_affinity(sense_food=sf)  # (B,C,H,W)
        self._redistribute(Aff)

        if self.has_food:
            self._food_step()
        self.frames += 1

    # ------------------------------------------------------------------ #
    #  Neighborhood helpers (roll-based, circular boundary)               #
    # ------------------------------------------------------------------ #
    def _neighbor_stack(self, field):
        """
        Returns (B,C,9,H,W): for each source position s, the value of `field`
        at neighbor s + delta_k (i.e. field rolled by -delta_k).
        """
        return torch.stack(
            [torch.roll(field, shifts=(-di, -dj), dims=(2, 3)) for (di, dj) in self._offsets],
            dim=2,
        )

    def _scatter(self, w):
        """
        Applies column-stochastic weights w (B,C,9,H,W, summing to 1 over the
        9-dim at each source) to redistribute mass, updating self.state.

        out_k[s] = rho_s * w_{s->k} is the mass leaving source s toward offset
        delta_k; it is deposited at destination s + delta_k (roll by +delta_k).
        """
        out = self.state.unsqueeze(2) * w  # (B,C,9,H,W)
        new = torch.zeros_like(self.state)
        for k, (di, dj) in enumerate(self._offsets):
            new = new + torch.roll(out[:, :, k], shifts=(di, dj), dims=(2, 3))
        self.state = new

    # ------------------------------------------------------------------ #
    #  Redistribution dispatch                                            #
    # ------------------------------------------------------------------ #
    def _redistribute(self, Aff):
        """
        Computes the 9 redistribution weights for the selected scheme and
        scatters the mass. Delta = A_n - A_s, shape (B,C,9,H,W).
        """
        # Semi-Lagrangian is a forward-push (4-cell bilinear scatter), not a
        # 9-stencil weight scheme, so it handles its own redistribution + diffusion.
        if self.redist_method == "semi-lagrangian":
            self._semi_lagrangian_step(Aff)
            return

        A_n = self._neighbor_stack(Aff)            # (B,C,9,H,W)
        delta = A_n - Aff.unsqueeze(2)             # Delta_{sn}

        m = self.redist_method
        if m == "MaCE":
            w = self._w_softmax(delta)
        elif m == "MaCE-iso":
            w = self._w_softmax(delta, iso=True)
        elif m == "sparsemax":
            w = self._w_sparsemax(self.b * delta)
        elif m == "entmax":
            w = self._w_entmax(delta)
        elif m == "winner":
            w = self._w_winner(delta)
        elif m == "uphill":
            w = self._family_b(self.lam * torch.clamp(delta, min=0.0))
        elif m == "eps-uphill":
            w = self._family_b(self.lam * (self.eps + torch.clamp(delta, min=0.0)))
        elif m == "metropolis":
            w = self._family_b((self.lam / 8.0) * torch.clamp(torch.exp(2 * self.b * delta), max=1.0))
        elif m == "barker":
            w = self._family_b((self.lam / 8.0) * torch.sigmoid(2 * self.b * delta))
        elif m == "exclusion":
            rho_n = self._neighbor_stack(self.state)
            cap = torch.clamp(1.0 - rho_n / self.rho_max, min=0.0)
            w = self._family_b((self.lam / 8.0) * torch.sigmoid(2 * self.b * delta) * cap)
        elif m == "sinkhorn":
            w = self._w_sinkhorn(delta)
        else:
            w = self._w_softmax(delta)

        # Universal isotropic diffusion knob: convex-blend the scheme's weights
        # toward the uniform box-average (1/9 over the 3x3, self included). A
        # convex combination of column-stochastic weights is column-stochastic,
        # so mass is still conserved exactly. d*box-average IS a fractional
        # heat-kernel step, adding tunable diffusion to schemes that have none
        # (uphill, sparsemax, winner, ...). At d=0 every scheme is unchanged;
        # for MaCE it simply stacks on top of the inherent (inevitable) diffusion.
        d = self.diffusion
        if d > 0.0:
            w = (1.0 - d) * w + d * (1.0 / 9.0)

        self._scatter(w)

    # ------------------------------------------------------------------ #
    #  Family A -- pull-normalized                                        #
    # ------------------------------------------------------------------ #
    def _w_softmax(self, delta, iso=False):
        """MaCE softmax: w ∝ [c_sn] exp(beta * Delta). Numerically stable."""
        e = torch.exp(self.b * delta)
        if iso:
            e = self._iso_c * e
        return e / e.sum(dim=2, keepdim=True)

    def _w_sparsemax(self, z):
        """
        Sparsemax (entmax alpha=2): w = [z - tau]_+ with tau set by the exact
        9-element water-filling. z = beta * Delta (sparsemax is shift-invariant).
        """
        K = z.shape[2]
        z_sorted, _ = torch.sort(z, dim=2, descending=True)
        rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype).view(1, 1, K, 1, 1)
        cssv = torch.cumsum(z_sorted, dim=2)
        cond = (1.0 + rng * z_sorted) > cssv          # support condition
        k_rho = (cond * rng).amax(dim=2, keepdim=True)  # largest valid k
        tau = (torch.gather(cssv, 2, (k_rho - 1).clamp(min=0).long()) - 1.0) / k_rho
        w = torch.clamp(z - tau, min=0.0)
        return w / w.sum(dim=2, keepdim=True).clamp(min=1e-12)

    def _w_entmax(self, delta):
        """
        Tsallis alpha-entmax: w = [(alpha-1) beta Delta - tau]_+^{1/(alpha-1)},
        tau fixed by sum_n w = 1. alpha->1 is softmax, alpha=2 is sparsemax.
        General alpha solved by per-cell bisection on tau (entmax is
        shift-invariant, so we work in z = (alpha-1) beta Delta).
        """
        alpha = self.alpha
        if abs(alpha - 1.0) < 1e-3:
            return self._w_softmax(delta)
        if abs(alpha - 2.0) < 1e-3:
            return self._w_sparsemax(self.b * delta)

        p = 1.0 / (alpha - 1.0)
        z = (alpha - 1.0) * self.b * delta
        zmax = z.amax(dim=2, keepdim=True)
        # g(tau) = sum [z - tau]_+^p is decreasing in tau.
        # tau = zmax => g = 0; tau = zmax - 1 => top term = 1 => g >= 1.
        tau_lo = zmax - 1.0
        tau_hi = zmax.clone()
        for _ in range(40):
            tau = 0.5 * (tau_lo + tau_hi)
            g = torch.clamp(z - tau, min=0.0).pow(p).sum(dim=2, keepdim=True)
            too_big = g > 1.0
            tau_lo = torch.where(too_big, tau, tau_lo)
            tau_hi = torch.where(too_big, tau_hi, tau)
        tau = 0.5 * (tau_lo + tau_hi)
        w = torch.clamp(z - tau, min=0.0).pow(p)
        return w / w.sum(dim=2, keepdim=True).clamp(min=1e-12)

    def _w_winner(self, delta):
        """Winner-take-all (beta -> inf): uniform over the argmax-affinity ties."""
        m = delta.amax(dim=2, keepdim=True)
        mask = (delta >= m - 1e-9).to(delta.dtype)
        return mask / mask.sum(dim=2, keepdim=True)

    # ------------------------------------------------------------------ #
    #  Family B -- push-rate with self-remainder                          #
    # ------------------------------------------------------------------ #
    def _family_b(self, f_off):
        """
        Builds column-stochastic weights from off-diagonal push rates f_off
        (the center entry is ignored): the self weight is the remainder
        w_{s->s} = 1 - sum_{n!=s} w_{s->n}. A per-cell limiter scales the
        off-diagonal block by min(1, 1/sum_off) so the remainder stays >= 0
        for any lambda (adaptive CFL cap, keeps conservation + positivity).
        """
        w = f_off.clone()
        w[:, :, self._center_k] = 0.0
        s_off = w.sum(dim=2, keepdim=True)
        scale = torch.clamp(1.0 / s_off.clamp(min=1e-12), max=1.0)
        w = w * scale
        s_off = w.sum(dim=2, keepdim=True)          # now <= 1 everywhere
        w[:, :, self._center_k] = (1.0 - s_off).squeeze(2)
        return w

    # ------------------------------------------------------------------ #
    #  Family D -- Sinkhorn (doubly-stochastic)                           #
    # ------------------------------------------------------------------ #
    def _w_sinkhorn(self, delta, sweeps=4):
        """
        Sinkhorn balancing: start from K_sn = exp(beta Delta) on N_s, then
        alternately normalize over n (outgoing rows) and over s (incoming
        columns) for a few sweeps, resisting over-concentration. Finish on a
        row normalization so the result is column-stochastic (conserves mass).
        """
        K = torch.exp(self.b * delta)
        for _ in range(sweeps):
            # Row normalization: outgoing mass from each source sums to 1.
            K = K / K.sum(dim=2, keepdim=True).clamp(min=1e-12)
            # Column normalization: total mass arriving at each destination = 1.
            col = torch.zeros_like(self.state)
            for k, (di, dj) in enumerate(self._offsets):
                col = col + torch.roll(K[:, :, k], shifts=(di, dj), dims=(2, 3))
            for k, (di, dj) in enumerate(self._offsets):
                dest_col = torch.roll(col, shifts=(-di, -dj), dims=(2, 3))  # col at s+delta_k
                K[:, :, k] = K[:, :, k] / dest_col.clamp(min=1e-12)
        # Final row normalization -> column-stochastic, exact conservation.
        return K / K.sum(dim=2, keepdim=True).clamp(min=1e-12)

    # ------------------------------------------------------------------ #
    #  Semi-Lagrangian -- conservative forward-push advection             #
    # ------------------------------------------------------------------ #
    def _semi_lagrangian_step(self, Aff):
        """
        Conservative (forward-push / cloud-in-cell) semi-Lagrangian advection,
        targeting the pure continuity equation  d rho/dt = -div(rho v),  with
        v = 2*beta*grad(A) -- the advective part of MaCE's continuous limit,
        WITHOUT the forced beta=0 Laplacian floor.

        Each cell is a parcel of mass rho_ij; move its center forward by v*dt
        (cell units, dx=1) and deposit its mass onto the 4 surrounding grid
        cells with bilinear (cloud-in-cell) weights. The 4 weights sum to 1 and
        the domain is periodic, so mass is conserved exactly; all weights and
        masses are >= 0, so positivity holds; there is no division by rho, so
        vacuum (rho=0) fronts are safe. No CFL *stability* limit (always
        conservative); keep |v|*dt < ~1 for accuracy.

        The 'diffusion' slider d adds the optional, separately-conservative
        5-point-Laplacian sub-step rho <- rho + (d/4)*lap(rho); d/4 <= 1/4 is
        the positivity bound. This is the *tunable* diffusion, decoupled from
        beta; set d=0 for pure advection.
        """
        B, C, H, W = self.state.shape
        dev = self.state.device
        N = B * C
        rho = self.state.reshape(N, H, W)
        A = Aff.reshape(N, H, W)

        # v = 2 beta grad(A), central differences, periodic. dim 1 = y/H, dim 2 = x/W.
        dAdx = (torch.roll(A, -1, dims=2) - torch.roll(A, 1, dims=2)) * 0.5
        dAdy = (torch.roll(A, -1, dims=1) - torch.roll(A, 1, dims=1)) * 0.5
        vx = 2.0 * self.b * dAdx
        vy = 2.0 * self.b * dAdy

        # Forward trajectory, displacement in cells (dx = 1).
        I = torch.arange(W, device=dev).view(1, 1, W)  # x / column index
        J = torch.arange(H, device=dev).view(1, H, 1)  # y / row index
        x_new = I + vx * self.sl_dt
        y_new = J + vy * self.sl_dt

        i0 = torch.floor(x_new)
        j0 = torch.floor(y_new)
        fx = x_new - i0
        fy = y_new - j0
        i0 = i0.long()
        j0 = j0.long()

        # Periodic wrap of the 2x2 target block.
        i0m, i1m = i0 % W, (i0 + 1) % W
        j0m, j1m = j0 % H, (j0 + 1) % H

        w00 = (1.0 - fx) * (1.0 - fy)  # -> (i0,   j0)
        w10 = fx * (1.0 - fy)          # -> (i0+1, j0)
        w01 = (1.0 - fx) * fy          # -> (i0,   j0+1)
        w11 = fx * fy                  # -> (i0+1, j0+1)

        # Scatter-add into a flat (N, H*W) buffer; linear index = j*W + i.
        out = torch.zeros(N, H * W, device=dev, dtype=rho.dtype)
        for jj, ii, ww in ((j0m, i0m, w00), (j0m, i1m, w10), (j1m, i0m, w01), (j1m, i1m, w11)):
            idx = (jj * W + ii).reshape(N, H * W)
            out.scatter_add_(1, idx, (rho * ww).reshape(N, H * W))
        rho = out.reshape(N, H, W)

        # Optional isotropic diffusion sub-step (5-point Laplacian), kappa = d/4.
        d = self.diffusion
        if d > 0.0:
            lap = (
                torch.roll(rho, 1, 1) + torch.roll(rho, -1, 1)
                + torch.roll(rho, 1, 2) + torch.roll(rho, -1, 2) - 4.0 * rho
            )
            rho = rho + (0.25 * d) * lap

        self.state = rho.reshape(B, C, H, W)

    # ------------------------------------------------------------------ #
    #  GUI / events                                                       #
    # ------------------------------------------------------------------ #
    def process_gui_change(self, component):
        super().process_gui_change(component)
        if component == self.method_dropdown:
            self.redist_method = self.method_dropdown.selected
        elif component == self.lambda_slider:
            self.lam = self.lambda_slider.value
        elif component == self.alpha_slider:
            self.alpha = self.alpha_slider.value
        elif component == self.eps_slider:
            self.eps = self.eps_slider.value
        elif component == self.diffusion_slider:
            self.diffusion = self.diffusion_slider.value
        elif component == self.sl_dt_slider:
            self.sl_dt = self.sl_dt_slider.value

    def process_event(self, event, camera=None):
        """
        LEFT/RIGHT: cycle the redistribution scheme (no state reset)
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                step = 1 if event.key == pygame.K_RIGHT else -1
                idx = (self.METHODS.index(self.redist_method) + step) % len(self.METHODS)
                self.redist_method = self.METHODS[idx]
                self.method_dropdown.selected = self.redist_method

    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of TestConservLenia.process_event

    def get_string_state(self):
        return super().get_string_state() + (
            f" | method: {self.redist_method}, lambda: {self.lam:.2f}, "
            f"alpha: {self.alpha:.2f}, diff: {self.diffusion:.2f}, sl_dt: {self.sl_dt:.3f}"
        )
