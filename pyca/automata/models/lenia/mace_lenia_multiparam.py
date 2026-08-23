import torch, torch.nn.functional as F
import pygame
import showtens
import math

from .mace_lenia_xchan import MaCELeniaXChan
from .lenia import Lenia
from .leniaparams import LeniaParams
from ...utils.noise_gen import perlin
from pyca.interface import MultiToggle, LabeledSlider, Button


class MaCELeniaMultiParam(MaCELeniaXChan):
    """
    MaCELenia where the growth-mixing weights are a *spatial* field that flows
    with the matter. Each pixel carries one of K 'species', a full weight vector
    that mixes the (C*k_mult) growth contributions. At every step each pixel
    inherits a species from its 3x3 neighborhood, sampled from a softmax over the
    incoming mass aggregated per species, so the parameters are advected and
    selected together with the mass (using the channel-averaged affinity).

    Optional cross-channel mixing (alpha>0, C>1) is inherited from MaCELeniaXChan
    and skipped entirely when alpha==0 to save compute.
    """

    def __init__(
        self,
        size,
        num_channels=3,
        k_mult=1,
        num_species=5,
        params=None,
        state_init=None,
        device="cpu",
        has_food=False,
        sense_food=False,
        interest_files=None,
        save_dir=".",
    ):
        """
        Args:
            size : tuple, (B,H,W) or (H,W) size of the automaton
            num_channels : int, number of channels
            k_mult : int, number of kernels per channel (source multiplier).
                Used only when generating default params (params is None).
            num_species : int, number of distinct parameter 'species' (K)
            params : dict/LeniaParams, parameters of the automaton
            state_init : tensor, initial state
            device : str, device to use
        """
        # Generate default params with the requested k_mult if none provided, so
        # k_mult>1 configurations (notably the 1-channel version) work out of the box.
        if params is None:
            batch_size = size[0] if len(size) == 3 else 1
            params = LeniaParams.default_gen(
                batch_size=batch_size,
                num_channels=num_channels,
                k_size=31,
                k_mult=k_mult,
                device=device,
            )

        self.num_species = num_species
        self.show_params = False
        self._param_temp = 1.0
        # Placeholders so _affinity_weights / set_init_* can detect "field not ready"
        # while the parent constructors run (they call set_init_circle / _compute_affinity).
        self.labels = None
        self.species = None

        super().__init__(
            size,
            num_channels=num_channels,
            params=params,
            state_init=state_init,
            has_food=has_food,
            sense_food=sense_food,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        # Cross-channel mixing off by default (saves compute); user can raise alpha.
        self.alpha = 0.0
        self.params["alpha"] = self.alpha
        self.alpha_slider.value = self.alpha

        # Build the species codebook + label field now that all shapes are known.
        self._init_param_field()

        # GUI components
        self.tog_param_view = MultiToggle(states=["View: State", "View: Param"], manager=self.manager)
        self.register_component(self.tog_param_view)
        self.k_mult_slider = LabeledSlider(
            min_value=1, max_value=6, manager=self.manager, title="k_mult",
            initial_value=self.k_mult, tick_size=1,
        )
        self.register_component(self.k_mult_slider)
        self.param_temp_slider = LabeledSlider(
            min_value=0.1, max_value=50.0, manager=self.manager, title="ParamTemp",
            initial_value=self._param_temp, tick_size=0.1,
        )
        self.register_component(self.param_temp_slider)
        self.k_slider = LabeledSlider(
            min_value=2, max_value=64, manager=self.manager, title="K species",
            initial_value=self.num_species, tick_size=1,
        )
        self.register_component(self.k_slider)
        self.grid_init_button = Button("Grid Species Init", manager=self.manager)
        self.register_component(self.grid_init_button)

    # ===================== Parameter field setup =====================

    def _init_param_field(self):
        """
        (Re)builds the K species codebook, the random RGB projection for the
        parameter view, and a fresh coherent label map.
        """
        B, C, k_mult, K = self.batch, self.C, self.k_mult, self.num_species
        S = C * k_mult

        # Codebook: K random, normalized mixing-weight vectors (B,K,S,C).
        species = torch.rand(B, K, S, C, device=self.device)
        species = species / species.sum(dim=2, keepdim=True).clamp_min(1e-6)  # normalize over sources
        self.species = species  # (B,K,S,C)

        # Random projection of a flattened weight vector (P=S*C) to RGB.
        self.param_proj = torch.randn(3, S * C, device=self.device)  # (3,P)
        self._refresh_species_colors()

        # Coherent spatial assignment of species to pixels.
        self._seed_labels()

    def _refresh_species_colors(self):
        """Precompute one stable RGB color per species: normalize(proj @ weight_vec)."""
        B, K = self.species.shape[0], self.num_species
        vecs = self.species.reshape(B, K, -1)  # (B,K,P)
        colors = torch.einsum("cp,bkp->bkc", self.param_proj, vecs)  # (B,K,3)
        cmin = colors.amin(dim=1, keepdim=True)
        cmax = colors.amax(dim=1, keepdim=True)
        self.species_colors = (colors - cmin) / (cmax - cmin).clamp_min(1e-6)  # (B,K,3) in [0,1]

    def _seed_labels(self):
        """Assigns each pixel to a species in coherent regions via low-freq perlin argmax."""
        K = self.num_species
        wavelength = max(8, min(self.h, self.w) // 3)
        noise = perlin(
            (self.batch, self.h, self.w), [wavelength, wavelength],
            num_channels=K, black_prop=0.0, device=self.device,
        )  # (B,K,H,W)
        self.labels = noise.argmax(dim=1).to(torch.long)  # (B,H,W)

    def _affinity_weights(self):
        """Spatial growth-mixing weights, gathered from the codebook by per-pixel label."""
        if self.labels is None or self.species is None:
            # Parent constructors call _compute_affinity before the field exists.
            return self.weights[..., None, None]
        B = self.batch
        batch_idx = torch.arange(B, device=self.device)[:, None, None]  # (B,1,1)
        theta = self.species[batch_idx, self.labels]  # (B,H,W,S,C)
        return theta.permute(0, 3, 4, 1, 2)  # (B,S,C,H,W)

    # ===================== Dynamics =====================

    def step(self):
        """Mass + parameter step, with optional cross-channel mixing."""
        Aff = self._mace_step(sense_food=self.sense_food)
        if self.has_food:
            self._food_step()
        if self.alpha != 0 and self.C > 1:
            self._cross_chan_step(Aff)  # only when actually mixing channels
        self.frames += 1

    def _mace_step(self, sense_food=False):
        """Runs the base MaCE mass redistribution, then flows the species labels."""
        x_old = self.state  # base step reassigns self.state, so this keeps the pre-update tensor
        Aff = super()._mace_step(sense_food=sense_food)  # (B,C,H,W), updates self.state
        self.labels = self._flow_labels(Aff, x_old)
        return Aff

    @torch.no_grad()
    def _flow_labels(self, Aff, x_old):
        """
        Advect + select the species labels. For each pixel, aggregate the incoming
        mass from its 3x3 neighborhood per species (these are the softmax logits),
        then sample a new species. Empty pixels keep their current label.

        Args:
            Aff : (B,C,H,W) affinity used for the mass step (pre-update).
            x_old : (B,C,H,W) state before the mass step (the mass that carried the params).
        """
        B, C, H, W = x_old.shape
        K = self.num_species
        eps = 1e-9

        # Channel-averaged affinity drives the parameter flow.
        Abar = Aff.mean(dim=1, keepdim=True)  # (B,1,H,W)
        E = torch.exp(self.b * Abar)  # (B,1,H,W)
        Zp = F.pad(E, (1, 1, 1, 1), mode="circular")
        Zp = F.unfold(Zp, kernel_size=(3, 3)).reshape(B, 9, H, W).sum(dim=1, keepdim=True)  # (B,1,H,W)
        m = x_old.sum(dim=1, keepdim=True)  # (B,1,H,W) total mass
        g = m / Zp.clamp_min(eps)  # (B,1,H,W) mass to give per source

        # Incoming mass from each of the 9 neighbors q -> p : a = E[p] * g[q]
        g_unf = F.pad(g, (1, 1, 1, 1), mode="circular")
        g_unf = F.unfold(g_unf, kernel_size=(3, 3)).reshape(B, 9, H, W)  # g[q]
        a = E * g_unf  # (B,9,H,W) (E[p] broadcast over the 9 sources)

        # Neighbor labels (same unfold ordering as the amounts).
        lab = self.labels.unsqueeze(1).float()  # (B,1,H,W)
        lab_unf = F.pad(lab, (1, 1, 1, 1), mode="circular")
        lab_unf = F.unfold(lab_unf, kernel_size=(3, 3)).reshape(B, 9, H, W).long()  # (B,9,H,W)

        # Aggregate incoming mass per species -> logits (B,K,H,W).
        logits = torch.zeros(B, K, H, W, device=self.device)
        logits.scatter_add_(1, lab_unf, a)
        counts = torch.zeros(B, K, H, W, device=self.device)
        counts.scatter_add_(1, lab_unf, torch.ones_like(a))

        # Species absent from the neighborhood cannot be inherited.
        logits = torch.where(counts > 0, logits, torch.full_like(logits, float("-inf")))

        # Softmax (temperature) over species, then Gumbel-max sampling.
        logits = self._param_temp * logits
        u = torch.rand_like(logits).clamp(eps, 1 - eps)
        gumbel = -torch.log(-torch.log(u))  # standard Gumbel(0,1)
        sampled = (logits + gumbel).argmax(dim=1)  # (B,H,W)

        # Only resample where mass actually arrives; keep old labels in empty space.
        incoming = a.sum(dim=1)  # (B,H,W)
        return torch.where(incoming > 1e-6, sampled, self.labels)

    # ===================== Param/shape changes =====================

    def update_params(self, params, k_size_override=None):
        super().update_params(params, k_size_override=k_size_override)
        # Rebuild the field if the source dimension or batch changed (e.g. loaded file, k_mult change).
        if getattr(self, "species", None) is not None:
            if self.species.shape[2] != self.C * self.k_mult or self.species.shape[0] != self.batch:
                self._init_param_field()

    def set_k_mult(self, k_mult):
        """Regenerates kernels/params for a new k_mult and rebuilds the species field."""
        k_mult = max(1, int(k_mult))
        if k_mult == self.k_mult:
            return
        params = LeniaParams.default_gen(
            batch_size=self.batch, num_channels=self.C, k_size=self.k_size,
            k_mult=k_mult, device=self.device,
        )
        self.update_params(params)  # sets self.k_mult, recomputes kernels/growth
        self._init_param_field()  # rebuild codebook (S=C*k_mult), projection, labels

    def set_num_species(self, K):
        """Regenerates the codebook with K species and re-seeds the label field."""
        K = max(1, int(K))
        if K == self.num_species:
            return
        self.num_species = K
        self._init_param_field()  # new K-vector codebook + projection + labels

    def set_init_grid(self, n=8, density=0.15):
        """
        Lays the species out as an n x n grid of squares (each a species drawn
        uniformly from the K species), over a low-density matter fill so the total
        starting mass stays small.

        Args:
            n : int, number of grid squares per side.
            density : float, scaling applied to the matter fill (lower = less mass).
        """
        self.set_init_perlin()  # textured matter over the whole window (also reseeds labels)
        self.state = self.state * density  # much less total mass
        if getattr(self, "species", None) is not None:
            self.labels = self._grid_labels(n)

    def _grid_labels(self, n=8):
        """Returns a (B,H,W) label map tiled as n x n blocks, each a random species."""
        B, H, W, K = self.batch, self.h, self.w, self.num_species
        block = torch.randint(0, K, (B, n, n), device=self.device)  # (B,n,n)
        # Nearest-neighbour upsample via even index maps (handles non-divisible H,W).
        row_idx = torch.arange(H, device=self.device) * n // H  # (H,) in [0,n)
        col_idx = torch.arange(W, device=self.device) * n // W  # (W,) in [0,n)
        labels = block[:, row_idx][:, :, col_idx]  # (B,H,W)
        return labels.to(torch.long)

    def set_init_fractal(self):
        super().set_init_fractal()
        if getattr(self, "species", None) is not None:
            self._seed_labels()

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if getattr(self, "species", None) is not None:
            self._seed_labels()

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if getattr(self, "species", None) is not None:
            self._seed_labels()

    # ===================== Interaction & rendering =====================

    def process_event(self, event, camera=None):
        """
        P: Toggle parameter view
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.show_params = not self.show_params
                self.tog_param_view.value = "View: Param" if self.show_params else "View: State"

    process_event.__doc__ = MaCELeniaXChan.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip("\n")

    def process_gui_change(self, component):
        super().process_gui_change(component)
        if component == self.tog_param_view:
            self.show_params = not self.tog_param_view.active
        elif component == self.param_temp_slider:
            self._param_temp = self.param_temp_slider.value
        elif component == self.k_mult_slider:
            self.set_k_mult(int(round(self.k_mult_slider.value)))
            self.k_mult_slider.value = self.k_mult
        elif component == self.k_slider:
            self.set_num_species(int(round(self.k_slider.value)))
            self.k_slider.value = self.num_species
        elif component == self.grid_init_button:
            self.set_init_grid()

    @torch.no_grad()
    def draw(self):
        """Draws the worldmap. In parameter view, tints the matter by its species color."""
        if not self.show_params:
            super().draw()
            return

        mass = self.state.sum(dim=1, keepdim=True).clamp(0, 1)  # (B,1,H,W)
        batch_idx = torch.arange(self.batch, device=self.device)[:, None, None]  # (B,1,1)
        col = self.species_colors[batch_idx, self.labels]  # (B,H,W,3)
        img = col.permute(0, 3, 1, 2) * mass  # (B,3,H,W) tinted by mass

        if (not self.show_all) and (not self.show_all_override):
            toshow = img[self.show_batch]  # (3,H,W)
        else:
            toshow = showtens.gridify(img, max_width=self.size[1] * 2, columns=int(math.sqrt(self.batch)))

        self._worldmap = torch.clamp(toshow, 0.0, 1.0)

    def get_string_state(self):
        view = "Param" if self.show_params else "State"
        return super().get_string_state() + f", view: {view}, K: {self.num_species}, k_mult: {self.k_mult}, pT: {self._param_temp:.1f}"


class MaCELeniaMultiParam1C(MaCELeniaMultiParam):
    """
    Single-channel MaCELeniaMultiParam. With C=1 the species mix the k_mult growth
    kernels of the lone channel (k_mult is adjustable in the GUI). Cross-channel
    mixing is inactive (only one channel), so it is always skipped.
    """

    def __init__(self, size, num_channels=1, k_mult=3, **kwargs):
        super().__init__(size, num_channels=num_channels, k_mult=k_mult, **kwargs)
