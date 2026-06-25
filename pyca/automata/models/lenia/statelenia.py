import math
import torch
import torch.nn.functional as F
import pygame

from .lenia import Lenia
from .leniaparams import LeniaParams
from pyca.interface import LabeledSlider, DropDown, Button

"""
    State-Lenia : a continuum ("monokinetic") MaCE model with an internal state.

    Each grid cell carries a *scalar density* rho(x) AND a *C-dimensional internal
    state* s(x) (a unit vector, displayed as an RGB colour). The dynamics is the
    exact gradient-descent / Vlasov system derived for MaCE with an evolving state,
    in its mass-conserving monokinetic closure:

        d rho/dt + div(rho u)            = 0                       (continuity)
        d  s /dt + (u . grad) s          = V_s                     (advect + react)

    integrated as the conservative pair (rho, m = rho s):

        d rho/dt + div(rho u)            = 0
        d  m /dt + div(u (x) m)          = rho V_s

    The interaction is the usual Lenia kernel K, *modulated* by a state-alignment
    factor g(s.s'). With a single-channel LeniaParams giving one kernel K and one
    Gaussian growth bump G (mu_G, sigma_G), the per-cell potential and the two
    velocities are

        U(x)        = sum_d  K(d)        g(s(x).s(x-d)) rho(x-d)          (scalar)
        grad1 U(x)  = sum_d  (gradK)(d)  g(s(x).s(x-d)) rho(x-d)          (in space, s held fixed)
        d_s U(x)    = sum_d  K(d)        g'(s(x).s(x-d)) s(x-d) rho(x-d)  (in state)

        u(x)   = - c_rep grad(phi * rho)  +  G'(U) grad1 U
        V_s(x) =   G'(U) d_s U            (then projected onto the tangent of s)

    so position descends the private energy E = R(x) - G(U) through space, and the
    state descends the SAME energy through the internal sphere -- the state channel
    is homeostatic in exactly the way the spatial channel is: a cell under-target
    (G'>0) both moves up-U and rotates s toward its aligned neighbours; over-target
    (G'<0) it does the opposite.

    Implementation notes
    --------------------
    * g is the alignment non-linearity (default relu; selectable in the GUI, easily
      extended). Only g linear makes the sums separable/FFT-able; for a general g the
      three sums (U, grad1 U, d_s U) are evaluated in one shot with F.unfold -- the
      whole k x k kernel integral is tensorized (no Python loop over offsets). Memory
      is O(B C k^2 H W), so keep k_size modest (default 25) on large grids.
    * The state vector s is constrained to the unit sphere: V_s is tangent-projected
      and s is renormalised after every push (free magnitude would drag U's scale).
    * Mass (sum rho) is conserved exactly by the cloud-in-cell forward push (bilinear
      weights sum to 1, periodic domain); the reaction only feeds m, never rho.
    * Uses the SAME LeniaParams as the rest of the zoo, but ALWAYS single-channel:
      it is NOT directly compatible with multi-channel MaCE param files (those get
      collapsed to their channel-0 self-interaction).
    """

class StateLenia(Lenia):

    # alignment non-linearity g and its derivative g'
    G_FUNCS = {
        "relu":   (lambda z: torch.clamp(z, min=0.0),      lambda z: (z > 0).to(z.dtype)),
        "linear": (lambda z: z,                            lambda z: torch.ones_like(z)),
        "relu^2": (lambda z: torch.clamp(z, min=0.0) ** 2, lambda z: 2.0 * torch.clamp(z, min=0.0)),
        "tanh":   (lambda z: torch.tanh(z),                lambda z: 1.0 - torch.tanh(z) ** 2),
    }
    G_NAMES = ["relu", "linear", "relu^2", "tanh"]

    def __init__(
        self,
        size,
        state_dim=3,
        dt=0.1,
        params=None,
        state_init=None,
        device="cpu",
        interest_files=None,
        save_dir=".",
    ):
        """
        Args:
            size : (B,H,W) or (H,W), size of the automaton (and #batches)
            state_dim : int, dimension C of the internal state vector s
            dt : float, reaction time-step (rate of the V_s update)
            params : LeniaParams / dict, single-channel Lenia params (collapsed if multi-channel)
            state_init : (B,1,H,W) tensor, optional initial density
            device : str
        """
        if len(size) == 2:
            size = (1,) + size
        self.state_dim = state_dim
        self.has_food = False  # never; here so the inherited resize() is happy

        # Force a single Lenia channel : one kernel K, one Gaussian growth G.
        params = self._coerce_single_channel(params, batch=size[0], device=device)

        super().__init__(
            size,
            dt=dt,
            num_channels=1,
            params=params,
            state_init=state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        self.show_batch = 0

        # --- alignment non-linearity ---
        self.g_name = "relu"

        # --- dynamics knobs ---
        # `flow` is the advection-speed gain: displacement = flow * u (cells/step).
        # It must be large because u = G'(U) grad1 U is intrinsically tiny (the Lenia
        # kernel is normalised to sum 1, so its spatial gradient is small).
        self.flow = 30.0       # advection speed gain
        self.c_rep = 1.0       # repulsion strength  (u -= c_rep grad(phi*rho))
        self.max_disp = 4.0    # safety clamp on |displacement| (cells)

        self.diffusion = 0.0   # optional isotropic density diffusion d in [0,1]

        # Fixed (non-GUI) parameters.
        self.sigma_rep = 4.0   # repulsion kernel width (cells)
        # The alignment factor g(s.s') <= 1 shrinks U ~4x below the bare Lenia
        # potential, so the Gaussian growth (tuned for the bare potential) would sit
        # in its dead tail (G' = 0 -> nothing moves). We rescale the growth setpoint
        # to U's actual band and floor sigma so G' stays alive at start-up.
        self.growth_shift = 0.35  # effective target mu_eff = growth_shift * mu
        self.sigma_floor = 0.12   # lower bound on the growth width

        # --- derived buffers (kernel weights + repulsion kernel + state field) ---
        self._build_kernel_weights()
        self._build_rep_kernel()
        self._kernel_ready = True  # subsequent update_params() may rebuild weights
        self._init_state_field()

        # --- GUI (only the knobs worth tuning) ---
        self.g_dropdown = DropDown(options=self.G_NAMES, manager=self.manager, starting_option=self.g_name)
        self.register_component(self.g_dropdown)
        self.flow_slider = LabeledSlider(
            min_value=0.0, max_value=100.0, manager=self.manager, title="flow speed",
            initial_value=self.flow, tick_size=1.0,
        )
        self.register_component(self.flow_slider)
        self.dt_slider = LabeledSlider(
            min_value=0.0, max_value=1.0, manager=self.manager, title="react dt",
            initial_value=self.dt, tick_size=0.01,
        )
        self.register_component(self.dt_slider)
        self.crep_slider = LabeledSlider(
            min_value=0.0, max_value=5.0, manager=self.manager, title="repulsion",
            initial_value=self.c_rep, tick_size=0.05,
        )
        self.register_component(self.crep_slider)
        self.diff_slider = LabeledSlider(
            min_value=0.0, max_value=1.0, manager=self.manager, title="diffusion",
            initial_value=self.diffusion, tick_size=0.01,
        )
        self.register_component(self.diff_slider)
        self.uniform_state_button = Button("Uniform State Field", manager=self.manager)
        self.register_component(self.uniform_state_button)
        self.add_kernel_button = Button("Add Kernel", manager=self.manager)
        self.register_component(self.add_kernel_button)
        self.remove_kernel_button = Button("Remove Kernel", manager=self.manager)
        self.register_component(self.remove_kernel_button)

    # ------------------------------------------------------------------ #
    #  Setup helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_single_channel(params, batch, device):
        """
        Returns a single-channel LeniaParams. If `params` is None, a fresh
        single-channel default is generated; if it carries more channels, only the
        channel-0 self-interaction (mu[0,0], sigma[0,0], K[0,0], ...) is kept.
        """
        if params is None:
            return LeniaParams(batch_size=batch, k_size=25, channels=1, device=device)

        if isinstance(params, LeniaParams):
            pd = dict(params.param_dict)
        else:
            pd = dict(params)

        nc = pd.get("num_channels", None)
        if nc == 1:
            return params if isinstance(params, LeniaParams) else LeniaParams(param_dict=pd, device=device)

        # Collapse to channel-0 self pair. (B, Ck, C[, R]) -> (B, 1, 1[, R])
        for key in ("mu", "sigma", "weights"):
            if key in pd and torch.is_tensor(pd[key]):
                pd[key] = pd[key][:, :1, :1].contiguous()
        for key in ("beta", "mu_k", "sigma_k", "k_coeffs", "k_harmonics", "g_coeffs", "g_harmonics"):
            if key in pd and torch.is_tensor(pd[key]):
                pd[key] = pd[key][:, :1, :1, ...].contiguous()
        pd["num_channels"] = 1
        pd["k_mult"] = 1
        return LeniaParams(param_dict=pd, device=device)

    def _build_kernel_weights(self):
        """
        Precomputes the per-offset weights of the U / grad1 U sums for EACH of the
        M = k_mult kernels, laid out to match F.unfold's patch order (row-major over
        the k x k window). Each is a (B, M, k*k, 1, 1) tensor, ready to broadcast (one
        kernel m at a time) against the unfolded neighbour stack (B, k*k, H, W).

        F.unfold returns, for output cell x, the neighbours at x + d with
        d = (i-p, j-p) (window index (i,j), p = k//2). So the weight that multiplies
        neighbour (i,j) is the kernel/kernel-gradient evaluated at x - x' = -d:

            U / d_s U weight :  K_bar(-d) =  K[i,j]      (K radial -> even)
            grad1 U weight   : (gradK_bar)(-d) = -(gradK)(d) = (-dK/dx, -dK/dy)[i,j]

        (the minus is the only place the unfold's x+d convention shows up). Each K is
        already normalised to sum 1 by Lenia, so each U_m lands in roughly [0,1].
        """
        K = self.k[:, :, 0]                                  # (B,M,k,k) -- M kernels, target chan 0
        dKi, dKj = torch.gradient(K, dim=(2, 3))             # d/d row (=y), d/d col (=x)
        B, M = K.shape[0], K.shape[1]
        kk = self.k_size * self.k_size

        # Row-major flatten (i*k + j) matches F.unfold's patch ordering exactly.
        self._kw = K.reshape(B, M, kk, 1, 1)                 # K_m[i,j]
        self._gkx = (-dKj).reshape(B, M, kk, 1, 1)           # grad1U_m x-weight (column)
        self._gky = (-dKi).reshape(B, M, kk, 1, 1)           # grad1U_m y-weight (row)

    def _build_rep_kernel(self):
        """Builds the normalised Gaussian repulsion kernel phi (size ~ 6 sigma_rep)."""
        sig = max(self.sigma_rep, 1e-3)
        r = max(1, int(math.ceil(3 * sig)))
        ax = torch.arange(-r, r + 1, device=self.device, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
        self._rep_kernel = (g / g.sum())[None, None]         # (1,1,kr,kr)
        self._rep_r = r

    def _init_state_field(self, uniform=False):
        """
        (Re)initialises the internal state s to unit vectors. With uniform=True the
        whole field is set to a single random direction (per batch) -- every cell
        shares the same state; otherwise each cell gets its own random direction.
        """
        if uniform:
            v = torch.randn(self.batch, self.state_dim, 1, 1, device=self.device)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
            self.s = v.expand(self.batch, self.state_dim, self.h, self.w).contiguous()
        else:
            s = torch.randn(self.batch, self.state_dim, self.h, self.w, device=self.device)
            self.s = s / s.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def update_params(self, params, k_size_override=None):
        # Base sets self.k / growth; rebuild our kernel-derived weights afterwards.
        super().update_params(params, k_size_override=k_size_override)
        if getattr(self, "_kernel_ready", False):
            self._build_kernel_weights()
            if self.s.shape[0] != self.batch or self.s.shape[1] != self.state_dim:
                self._init_state_field()

    # ------------------------------------------------------------------ #
    #  Add / remove kernels (k_mult) -- each kernel has its own growth     #
    # ------------------------------------------------------------------ #
    # The number of kernels is the LeniaParams `k_mult` (here, since C=1, it is just
    # the count of kernels). Each kernel carries its own (mu, sigma) growth and its
    # own weight; the per-kernel growth contributions are weights-averaged (see
    # _assemble_fields). Adding keeps the existing kernels and appends fresh random
    # ones; removing drops the last ones. The current rho/s world is preserved.

    def add_kernel(self):
        self._set_num_kernels(self.k_mult + 1)

    def remove_kernel(self):
        if self.k_mult > 1:
            self._set_num_kernels(self.k_mult - 1)

    def _set_num_kernels(self, m_new):
        m_new = max(1, int(m_new))
        m_old = self.k_mult
        if m_new == m_old:
            return

        # per-kernel tensors live on dim 1 (length C*k_mult == k_mult since C == 1)
        per_kernel = ("mu", "sigma", "weights", "beta", "mu_k", "sigma_k")
        pd = dict(self.params.param_dict)
        # This model always uses the standard Gaussian growth (mu/sigma); drop any
        # arbitrary-function tensors so they cannot clash with the new k_mult.
        for key in ("k_coeffs", "k_harmonics", "k_rescale", "k_clip",
                    "g_coeffs", "g_harmonics", "g_rescale", "g_clip"):
            pd.pop(key, None)

        if m_new > m_old:
            extra = LeniaParams.default_gen(
                batch_size=self.batch, num_channels=1, k_size=self.k_size,
                k_mult=(m_new - m_old), device=self.device,
            )
            for key in per_kernel:
                pd[key] = torch.cat([self.params[key], extra[key]], dim=1)
        else:
            for key in per_kernel:
                pd[key] = self.params[key][:, :m_new].contiguous()

        pd["k_mult"] = m_new
        pd["num_channels"] = 1
        self.g_arbi = False
        self.k_arbi = False
        self.update_params(LeniaParams(param_dict=pd, device=self.device))

    # ------------------------------------------------------------------ #
    #  Growth + alignment                                                 #
    # ------------------------------------------------------------------ #
    def _growth_deriv(self, U, mu, sigma):
        """
        G'(U) for the Gaussian bump G(U) = 2 exp(-(U-mu)^2/2sigma^2) - 1, with the
        target/width rescaled into U's actual (alignment-shrunk) band so G' is live.
        mu, sigma are this kernel's growth parameters, shape (B,).
        """
        mu = mu.view(-1, 1, 1) * self.growth_shift                    # (B,1,1)
        sig = sigma.view(-1, 1, 1).clamp(min=self.sigma_floor)
        bump = torch.exp(-((U - mu) ** 2) / (2 * sig ** 2))
        return -2.0 * (U - mu) / (sig ** 2) * bump                    # (B,H,W)

    @property
    def _g(self):
        return self.G_FUNCS[self.g_name][0]

    @property
    def _gp(self):
        return self.G_FUNCS[self.g_name][1]

    # ------------------------------------------------------------------ #
    #  Field assembly -- tensorized via unfold, averaged over kernels      #
    # ------------------------------------------------------------------ #
    def _assemble_fields(self):
        """
        Returns the growth-driven velocity (vx, vy) (B,H,W) and the (un-projected)
        state reaction Vs (B,C,H,W), averaged over the M = k_mult kernels.

        Each kernel m has its OWN growth G_m, so it contributes G_m'(U_m) grad1 U_m to
        the velocity and G_m'(U_m) d_s U_m to the reaction; we take the weights-average
        over kernels (the Lenia `weights`, normalised to sum 1):

            vx, vy = sum_m w_m G_m'(U_m) grad1 U_m
            Vs     = sum_m w_m G_m'(U_m) d_s U_m

        The expensive part -- the F.unfold neighbour gather and the alignment factor
        g(s(x).s(x+d)) -- is kernel-independent and done once; only the cheap weighted
        k*k-reductions are repeated per kernel (M is small). Memory is O(B C k^2 H W).
        """
        rho = self.state                                     # (B,1,H,W)
        s = self.s                                           # (B,C,H,W)
        B, C, H, W = s.shape
        k = self.k_size
        p = k // 2
        g, gp = self._g, self._gp

        # Neighbour stacks (circular boundary). Patch order matches the kernel weights.
        s_unf = F.unfold(F.pad(s, (p, p, p, p), mode="circular"), kernel_size=k)
        s_unf = s_unf.reshape(B, C, k * k, H, W)             # s(x+d)        (B,C,k2,H,W)
        rho_unf = F.unfold(F.pad(rho, (p, p, p, p), mode="circular"), kernel_size=k)
        rho_unf = rho_unf.reshape(B, k * k, H, W)            # rho(x+d)      (B,k2,H,W)

        sd = (s.unsqueeze(2) * s_unf).sum(dim=1)             # s(x).s(x+d)   (B,k2,H,W)
        q = g(sd) * rho_unf                                  # g(.)  rho(x+d) (B,k2,H,W)
        qg = gp(sd) * rho_unf                                # g'(.) rho(x+d) (B,k2,H,W)

        vx = torch.zeros(B, H, W, device=self.device)
        vy = torch.zeros(B, H, W, device=self.device)
        Vs = torch.zeros(B, C, H, W, device=self.device)
        for m in range(self.k_mult):
            kw = self._kw[:, m]                              # (B,k2,1,1)
            U_m = (kw * q).sum(dim=1)                        # (B,H,W)
            gUx_m = (self._gkx[:, m] * q).sum(dim=1)         # (B,H,W)
            gUy_m = (self._gky[:, m] * q).sum(dim=1)
            dsU_m = ((kw * qg).unsqueeze(1) * s_unf).sum(dim=2)   # (B,C,H,W)

            Gp_m = self._growth_deriv(U_m, self.mu[:, m, 0], self.sigma[:, m, 0])  # (B,H,W)
            wGp = self.weights[:, m, 0].view(B, 1, 1) * Gp_m      # (B,H,W)

            vx = vx + wGp * gUx_m
            vy = vy + wGp * gUy_m
            Vs = Vs + wGp.unsqueeze(1) * dsU_m

        return vx, vy, Vs

    def _rep_grad(self, rho):
        """grad(phi * rho) (periodic), returns (gx, gy), each (B,1,H,W)."""
        r = self._rep_r
        rp = F.pad(rho, (r, r, r, r), mode="circular")
        sm = F.conv2d(rp, self._rep_kernel)                  # (B,1,H,W)
        gx = (torch.roll(sm, -1, dims=3) - torch.roll(sm, 1, dims=3)) * 0.5
        gy = (torch.roll(sm, -1, dims=2) - torch.roll(sm, 1, dims=2)) * 0.5
        return gx, gy

    # ------------------------------------------------------------------ #
    #  Conservative semi-Lagrangian forward push (cloud-in-cell)          #
    # ------------------------------------------------------------------ #
    def _sl_push(self, field, ux, uy, dt):
        """
        Forward-push every channel of `field` (B,Ch,H,W) by the SHARED velocity
        (ux,uy) (B,H,W), scattering each parcel onto the 4 surrounding cells with
        bilinear weights (periodic). Mass per channel conserved exactly.
        """
        B, Ch, H, W = field.shape
        dev = field.device
        I = torch.arange(W, device=dev).view(1, 1, W)        # x / column
        J = torch.arange(H, device=dev).view(1, H, 1)        # y / row

        dx = torch.clamp(ux * dt, -self.max_disp, self.max_disp)
        dy = torch.clamp(uy * dt, -self.max_disp, self.max_disp)
        x_new = I + dx
        y_new = J + dy

        i0 = torch.floor(x_new)
        j0 = torch.floor(y_new)
        fx = x_new - i0
        fy = y_new - j0
        i0 = i0.long()
        j0 = j0.long()
        i0m, i1m = i0 % W, (i0 + 1) % W
        j0m, j1m = j0 % H, (j0 + 1) % H

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        out = torch.zeros(B, Ch, H * W, device=dev, dtype=field.dtype)
        fflat = field.reshape(B, Ch, H * W)
        for jj, ii, ww in ((j0m, i0m, w00), (j0m, i1m, w10), (j1m, i0m, w01), (j1m, i1m, w11)):
            idx = (jj * W + ii).reshape(B, 1, H * W).expand(B, Ch, H * W)
            out.scatter_add_(2, idx, fflat * ww.reshape(B, 1, H * W))
        return out.reshape(B, Ch, H, W)

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self):
        rho = self.state                                     # (B,1,H,W)
        s = self.s                                           # (B,C,H,W)

        # Growth-driven velocity (vx,vy) and raw state reaction Vs, averaged over kernels.
        vx, vy, Vs = self._assemble_fields()

        # --- state reaction V_s, projected onto the tangent of s ---
        Vs = Vs - (s * Vs).sum(dim=1, keepdim=True) * s      # (I - s s^T) Vs

        # --- velocity u (x=column, y=row) ---
        ux = vx
        uy = vy
        if self.c_rep > 0.0:
            gx, gy = self._rep_grad(rho)
            ux = ux - self.c_rep * gx[:, 0]
            uy = uy - self.c_rep * gy[:, 0]

        # --- conservative pair (rho, m = rho s); reaction feeds m only ---
        m = rho * s + self.dt * rho * Vs                     # (B,C,H,W)
        stacked = torch.cat([rho, m], dim=1)                 # (B,1+C,H,W)
        pushed = self._sl_push(stacked, ux, uy, self.flow)
        rho_new = pushed[:, :1]
        m_new = pushed[:, 1:]

        # --- optional isotropic diffusion (same for rho and m -> s preserved) ---
        if self.diffusion > 0.0:
            both = torch.cat([rho_new, m_new], dim=1)
            lap = (
                torch.roll(both, 1, 2) + torch.roll(both, -1, 2)
                + torch.roll(both, 1, 3) + torch.roll(both, -1, 3) - 4.0 * both
            )
            both = both + (0.25 * self.diffusion) * lap
            rho_new, m_new = both[:, :1], both[:, 1:]

        # --- recover the unit state ---
        # s = m / |m| where there is state-momentum to define a direction; where there
        # is essentially none (vacuum, or near-exact antipodal cancellation) we KEEP the
        # previous direction `s` instead of letting it collapse to 0. This keeps the whole
        # field on the unit sphere, so the affinity at empty cells, U = sum K g(s.s') rho',
        # reflects the actual neighbourhood (with s=0 it was forced to 0 via g(0)=0), and
        # mass-bearing cells that hit cancellation are not left stuck colourless.
        norm = m_new.norm(dim=1, keepdim=True)               # (B,1,H,W)
        s_new = m_new / norm.clamp(min=1e-9)
        self.s = torch.where(norm > 1e-6, s_new, s)          # `s` is the pre-step state
        self.state = rho_new.clamp(min=0.0)

        self.frames += 1

    # ------------------------------------------------------------------ #
    #  State (re)initialisation -- also reseed s                          #
    # ------------------------------------------------------------------ #
    def set_init_fractal(self):
        super().set_init_fractal()
        if hasattr(self, "s"):
            self._init_state_field()

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if hasattr(self, "s"):
            self._init_state_field()

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if hasattr(self, "s"):
            self._init_state_field()

    def resize(self, new_size):
        super().resize(new_size)
        self.s = F.interpolate(self.s, size=new_size, mode="bilinear", align_corners=False)
        self.s = self.s / self.s.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def _affect_matter(self, x, y, add=True):
        super()._affect_matter(x, y, add)
        if add:
            add_rad = self.k_size / 2.0
            mask = (self.X - x) ** 2 + (self.Y - y) ** 2 < add_rad ** 2  # (H,W)
            rnd = torch.randn(self.batch, self.state_dim, self.h, self.w, device=self.device)
            rnd = rnd / rnd.norm(dim=1, keepdim=True).clamp(min=1e-9)
            self.s[:, :, mask] = rnd[:, :, mask]

    # ------------------------------------------------------------------ #
    #  GUI / events                                                       #
    # ------------------------------------------------------------------ #
    def process_gui_change(self, component):
        super().process_gui_change(component)
        if component == self.g_dropdown:
            self.g_name = self.g_dropdown.selected
        elif component == self.flow_slider:
            self.flow = self.flow_slider.value
        elif component == self.dt_slider:
            self.dt = self.dt_slider.value
        elif component == self.crep_slider:
            self.c_rep = self.crep_slider.value
        elif component == self.diff_slider:
            self.diffusion = self.diff_slider.value
        elif component == self.uniform_state_button:
            self._init_state_field(uniform=True)
        elif component == self.add_kernel_button:
            self.add_kernel()
        elif component == self.remove_kernel_button:
            self.remove_kernel()

    def process_event(self, event, camera=None):
        """
        LEFT/RIGHT: cycle the alignment non-linearity g
        PLUS/MINUS: show next / previous batch
        ] / [ : add / remove a kernel (each kernel has its own growth)
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                step = 1 if event.key == pygame.K_RIGHT else -1
                idx = (self.G_NAMES.index(self.g_name) + step) % len(self.G_NAMES)
                self.g_name = self.G_NAMES[idx]
                self.g_dropdown.selected = self.g_name
            if event.key in (pygame.K_KP_PLUS, pygame.K_PLUS):
                self.show_batch = (self.show_batch + 1) % self.batch
            if event.key in (pygame.K_KP_MINUS, pygame.K_MINUS):
                self.show_batch = (self.show_batch - 1) % self.batch
            if event.key == pygame.K_RIGHTBRACKET:
                self.add_kernel()
            if event.key == pygame.K_LEFTBRACKET:
                self.remove_kernel()

    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip("\n")

    def get_string_state(self):
        mass = self.state.sum().item()
        return super().get_string_state() + (
            f" | mass: {mass:.1f}, g: {self.g_name}, C: {self.state_dim}, "
            f"kernels: {self.k_mult}, flow: {self.flow:.0f}, c_rep: {self.c_rep:.2f}, "
            f"batch: {self.show_batch}"
        )

    # ------------------------------------------------------------------ #
    #  Drawing : RGB = (s/2 + 1/2) * rho                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def draw(self):
        b = self.show_batch % self.batch
        rho = self.state[b]                                  # (1,H,W)
        s = self.s[b]                                        # (C,H,W)

        if self.state_dim == 1:
            col = (s * 0.5 + 0.5).repeat(3, 1, 1)
        elif self.state_dim == 2:
            col = torch.cat([s * 0.5 + 0.5, torch.zeros_like(s[:1])], dim=0)
        else:
            col = s[:3] * 0.5 + 0.5                           # (3,H,W), direction -> colour

        rgb = col * rho                                      # darken by density
        if self.display_kernel:
            rgb = self._draw_kernel(rgb)
        self._worldmap = torch.clamp(rgb, 0.0, 1.0)
