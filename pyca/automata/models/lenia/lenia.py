import torch, torch.nn.functional as F
from torchenhanced import DevModule
from ...utils.noise_gen import perlin, perlin_fractal
from .leniaparams import LeniaParams
from .funcgen import ArbitraryFunction
from .utils import create_smooth_circular_mask
import random, pygame
from ...automaton import Automaton
from pathlib import Path
from copy import deepcopy
from pyca.interface import Toggle, Button

class Lenia(DevModule, Automaton):
    """
    Multi-channel lenia automaton. A multi-colored GoL-inspired continuous automaton. Introduced by Bert Chan.   
    """

    def __init__(
        self,
        size,
        dt=0.1,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        interest_files=None,
        save_dir=".",
    ):
        """
        Initializes automaton.

        Args :
            size : (B,H,W) or (H,W) of ints, size of the automaton and number of batches
            dt : time-step used when computing the evolution of the automaton
            num_channels : int, number of channels (C) in the automaton
            params : LeniaParams class, or dict of parameters containing the following
                keys-values :
                'k_size' : odd int, size of kernel used for computations
                'mu' : (B,C,C) tensor, mean of growth functions
                'sigma' : (B,C,C) tensor, standard deviation of the growth functions
                'beta' :  (B,C,C, # of rings) float, max of the kernel rings
                'mu_k' : (B,C,C, # of rings) [0,1.], location of the kernel rings
                'sigma_k' : (B,C,C, # of rings) float, standard deviation of the kernel rings
                'weights' : (B,C,C) float, weights for the growth weighted sum
            or :
                'k_size' : odd int, size of kernel used for computations
                'k_coeffs' : (B,C,C, # of harmonics) float, coefficients for the kernel harmonics
                'k_harmonics' : (B,C,C # of harmonics) float, values of harmonics
                'g_coeffs' : (B,C,C, # of harmonics) float, coeffs for growth harmonics
                'g_harmonics' : (B,C,C, # of harmonics) float, values of growth harmonics
            state_init : (B,C,H,W) tensor, initial state of the automaton. If None, will be set to a circle of perlin noise
            interest_files : str or Path, path to a folder containing interesting states to load
            save_dir : str, directory to save interesting states to
            device : str, device
        """
        DevModule.__init__(self)
        if(len(size) == 2):
            size = (1,)+size
        Automaton.__init__(self, size[1:])

        self.to(device)

        self.batch = size[0]
        self.h, self.w = size[1:]
        self.C = num_channels

        if params is None:
            # Generates random parameters
            self.params = LeniaParams(batch_size=self.batch, k_size=31, channels=self.C, device=device)
        elif isinstance(params, dict):
            self.params = LeniaParams(param_dict=params, device=device)
        else:
            self.params = params

        self.k_size = self.params["k_size"]  # kernel size
        self.k_mult = self.params.param_dict.get("k_mult", 1)  # number of kernels (same for all)

        self.register_buffer("state", torch.rand((self.batch, self.C, self.h, self.w)))

        if state_init is None:
            self.set_init_circle()
        else:
            self.state = state_init.to(self.device)  # Specific init

        self.dt = dt

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer("mu", self.params["mu"])  # mean of the growth functions (B,C,C)
        self.register_buffer(
            "sigma", self.params["sigma"]
        )  # standard deviation of the growths functions (B,C,C)
        self.register_buffer("beta", self.params["beta"])  # max of the kernel rings (B,C,C, # of rings)
        self.register_buffer("mu_k", self.params["mu_k"])  # mean of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer(
            "sigma_k", self.params["sigma_k"]
        )  # standard deviation of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer(
            "weights", self.params["weights"]
        )  # raw weigths for the growth weighted sum (B,C,C)
        self.register_buffer("kernel", torch.zeros((self.k_size, self.k_size)))

        # For the generation, whether to use the arbitrary function or not
        self.g_arbi = False
        self.k_arbi = False
        self.update_params(self.params)

        # For interactivity and visualization
        self.display_kernel = False
        self.save_dir = save_dir
        if interest_files is not None:
            self.interest_files = [file_path.as_posix() for file_path in Path(interest_files).rglob("*.pt")]
        else:
            self.interest_files = None
        self.chosen_interesting = 0

        self.frames = 0

        self.add_speed = 5.0


        # GUI components
        self.reset_button = Button('Re-initialize')
        self.register_component(self.reset_button)
        self.cool_rule = Button('Next Interesting Rule')
        self.register_component(self.cool_rule)
        self.random_rule = Button('Re-roll Rule')
        self.register_component(self.random_rule)
        self.wipe_button = Button('Wipe State')
        self.register_component(self.wipe_button)
        self.toggle_rand_type = Toggle('Smart Random Rule', 'Uniform Random Rule')
        self.register_component(self.toggle_rand_type)
        self.toggle_add_remove = Toggle('Click to Add', 'Click to Remove')
        self.register_component(self.toggle_add_remove)
        

    def update_params(self, params: LeniaParams, k_size_override=None):
        """
        Updates the Lenia parameters (carefully).
        Changes batch size to match the one of provided params (take mu as reference)

        Args:
            params : LeniaParams
            k_size_override : int, if provided, will override the kernel size stored in params
        """
        self.frames = 0
        if isinstance(params, LeniaParams):
            params = params.param_dict

        self.mu = params.get("mu", self.mu)
        self.sigma = params.get("sigma", self.sigma)
        self.beta = params.get("beta", self.beta)
        self.mu_k = params.get("mu_k", self.mu_k)
        self.sigma_k = params.get("sigma_k", self.sigma_k)
        self.weights = params.get("weights", self.weights)
        self.k_size = params.get("k_size", self.k_size)  # kernel sizes (same for all)
        self.k_mult = params.get("k_mult", self.k_mult)  # number of kernels (same for all)
        self.g_arbi = params.get("g_arbi", self.g_arbi)  # whether to use arbitrary function for growth
        self.k_arbi = params.get("k_arbi", self.k_arbi)  # whether to use arbitrary function for kernel

        if k_size_override is not None:
            self.k_size = k_size_override

        if self.k_size % 2 == 0:
            self.k_size += 1
            print(f"Increased even kernel size to {self.k_size} to be odd")

        if "state" in params:
            self._load_state(params["state"])  #
            del params["state"]  # Remove state from params

        self.params = LeniaParams(param_dict=params, device=self.device)

        self.batch = self.params.batch_size

        if self.k_arbi and not "k_harmonics" in self.params:
            ## Translate usual kernel to Arbitrary since no arbi data

            transl_params = self.params.to_arbi_params(
                lenia_params=self.params, device=self.device
            )  # translate
            self.params["k_harmonics"] = transl_params["k_harmonics"]
            self.params["k_coeffs"] = transl_params["k_coeffs"]
            self.params["k_rescale"] = transl_params["k_rescale"]

        if self.g_arbi and not "g_harmonics" in self.params:
            ## Translate usual growth to Arbitrary since no arbi data
            transl_params = self.params.to_arbi_params(
                lenia_params=self.params, device=self.device
            )  # translate
            self.params["g_harmonics"] = transl_params["g_harmonics"]
            self.params["g_coeffs"] = transl_params["g_coeffs"]
            self.params["g_rescale"] = transl_params["g_rescale"]
            self.params["g_clip"] = transl_params["g_clip"]

        self.k = self.compute_kernel()  # (B,C,C,k_size,k_size)
        self.growth = self.compute_growth()  # growth function, callable

        self.fft_kernel = self.kernel_to_fft(self.k)  # (B,C,C,h,w)

    def resize(self, new_size):
        """
        Resizes the automaton world, and recomputes the fft_kernel to match.

        Args:
            new_size : (B,H,W) of ints, new size of the automaton and number of batches
        """
        super().resize(new_size)
        self.k = self.compute_kernel()
        self.fft_kernel = self.kernel_to_fft(self.k)  # (B,C,C,h,w)
        self.state = F.interpolate(self.state, size=new_size, mode="bilinear", align_corners=False)
        if self.has_food:
            self.food_channel = F.interpolate(
                self.food_channel, size=new_size, mode="bilinear", align_corners=False
            )

    def set_init_fractal(self):
        """
        Sets the initial state of the automaton using fractal perlin noise.
        Max wavelength is k_size*1.5, chosen a bit randomly
        """
        self.frames = 0
        self.state = perlin_fractal(
            (self.batch, self.h, self.w),
            int(self.k_size * 1.5),
            device=self.device,
            black_prop=0.25,
            num_channels=self.C,
            persistence=0.4,
        )

    def set_init_perlin(self, wavelength=None):
        """
        Sets initial state using one-wavelength perlin noise.
        Default wavelength is 2*K_size
        """
        self.frames = 0
        if not wavelength:
            wavelength = self.k_size
        self.state = perlin(
            (self.batch, self.h, self.w),
            [wavelength] * 2,
            device=self.device,
            num_channels=self.C,
            black_prop=0.25,
        )

    def set_init_circle(self, fractal=False, radius=None):
        """
        Creates a circle of perlin noise in the center of the world.

        Args:
            fractal : bool, whether to use fractal perlin noise or not
            radius : int, radius of the circle. If None, will be set to 3*k_size
        """
        self.frames = 0
        if radius is None:
            radius = self.k_size * 3
        if fractal:
            self.state = perlin_fractal(
                (self.batch, self.h, self.w),
                int(self.k_size * 1.5),
                device=self.device,
                black_prop=0.25,
                num_channels=self.C,
                persistence=0.4,
            )
        else:
            self.state = perlin(
                (self.batch, self.h, self.w),
                [self.k_size] * 2,
                device=self.device,
                num_channels=self.C,
                black_prop=0.25,
            )
        X, Y = torch.meshgrid(
            torch.linspace(-self.h // 2, self.h // 2, self.h, device=self.device),
            torch.linspace(-self.w // 2, self.w // 2, self.w, device=self.device),
        )
        R = torch.sqrt(X**2 + Y**2)
        self.state = torch.where(R < radius, self.state, torch.zeros_like(self.state, device=self.device))

    def kernel_slice(self, r):
        """
        Given a distance matrix r, computes the kernel of the automaton.
        In other words, compute the kernel 'cross-section' since we always assume
        rotationally symmetric kernel

        Args :
        r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None, None]  # (1,1, 1, 1, k_size, k_size)
        r = r.expand(
            self.batch, self.C * self.k_mult, self.C, self.mu_k.shape[3], -1, -1
        )  # (B,C,C,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None]  # (B,C,C,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]  # (B,C,C,#of rings,1,1)

        K = torch.exp(-(((r - mu_k) / sigma_k) ** 2) / 2)  # (B,C,C,#of rings,k_size,k_size)

        beta = self.beta[..., None, None]  # (B,C,C,#of rings,1,1)
        K = torch.sum(beta * K, dim=3)  #

        return K  # (B,C,C,k_size, k_size)

    def compute_kernel(self):
        """
        Computes the kernel given the current parameters. Uses in priority
        random fourier function if parameters are provided, else uses the standard way.

        Returns : Tensor of shape (B,C,C,k_size,k_size), the full set of kernels
        """
        xyrange = torch.linspace(-1, 1, self.k_size).to(self.device)

        X, Y = torch.meshgrid(
            xyrange, xyrange, indexing="xy"
        )  # (k_size,k_size),  axis directions is x increasing to the right, y increasing to the bottom
        r = torch.sqrt(X**2 + Y**2)  # (k_size,k_size)

        if self.k_arbi:
            assert "k_coeffs" in self.params.param_dict, "k_coeffs not in params"
            harmonics = self.params["k_harmonics"].reshape(
                self.batch * self.C * self.k_mult * self.C, -1
            )  # (B*C*C,# of harmonics)
            coeffs = self.params["k_coeffs"].reshape(
                self.batch * self.C * self.k_mult * self.C, -1
            )  # (B*C*C,# of harmonics)
            ranges = torch.tensor([0.0, 1.0], device=self.device)[None, :].expand(
                self.batch * self.C * self.k_mult * self.C, -1
            )
            arbi = ArbitraryFunction(
                coefficients=coeffs,
                harmonics=harmonics,
                ranges=ranges,
                rescale=self.params["k_rescale"],
                clips_min=0.0,
                device=self.device,
            )
            K = arbi(
                r[None].expand(self.batch * self.C * self.C * self.k_mult, -1, -1)
            )  # (BCC,k_size,k_size)
            K = K.reshape(self.batch, self.C * self.k_mult, self.C, self.k_size, self.k_size)
            K = create_smooth_circular_mask(K, self.k_size // 2)
        else:
            K = self.kernel_slice(r)  # (B,C*k_mult,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim=(-1, -2), keepdim=True)  # (B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed < 1e-6, 1, summed)
        K /= summed

        return K  # (B,C,C,k_size,k_size)

    def kernel_to_fft(self, K):
        """
        Computed the fft of the kernel correctly padded to compute convolutions with the world.

        Args:
            K : (B,C*k_mult,C,k_size,k_size), kernel to compute the fft of
        """
        # Pad kernel to match image size
        # For some reason, pad is left;right, top;bottom, (so W,H)
        K = F.pad(K, [0, (self.w - self.k_size)] + [0, (self.h - self.k_size)])  # (B,C*k_mult,C,h,w)

        # Center the kernel on the top left corner for fft
        K = K.roll((-(self.k_size // 2), -(self.k_size // 2)), dims=(-1, -2))  # (B,C*k_mult,C,h,w)

        K = torch.fft.fft2(K)  # (B,C*k_mult,C,h,w)

        return K  # (B,C*k_mult,C,h,w)

    def compute_growth(self):
        """
        Constructs the growth function given current parameters.
        By default, uses random fourier sampling if the necessary parameters are defined

        Returns : callable, growth function that takes as input the result of the convolution (B,C*k_mult,C,H,W)
        """

        if self.g_arbi:
            assert "g_coeffs" in self.params.param_dict, "g_coeffs not in params, but g_arbi is True"
            # Use ArbitraryFunction
            coeffs = self.params["g_coeffs"].reshape(
                self.batch * self.C * self.k_mult * self.C, -1
            )  # (B*C*C,# of harmonics)
            harmonics = self.params["g_harmonics"].reshape(
                self.batch * self.C * self.k_mult * self.C, -1
            )  # (B*C*C,# of harmonics)
            ranges = torch.tensor([0.0, 2.0], device=self.device)[None, :].expand(
                self.batch * self.C * self.k_mult * self.C, -1
            )

            arbi = ArbitraryFunction(
                coefficients=coeffs,
                harmonics=harmonics,
                ranges=ranges,
                rescale=self.params["g_rescale"],
                clips_min=self.params["g_clip"],
                device=self.device,
            )

            def growth(u):
                B, Ck, C, H, W = u.shape
                u = u.reshape(B * Ck * C, H, W)
                out = arbi(u)
                return out.reshape(B, Ck, C, H, W)

            return growth
        else:
            mu = self.mu[..., None, None]  # (B,C*k_mult,C,1,1)
            sigma = self.sigma[..., None, None]  # (B,C*k_mult,C,1,1)

            def growth(u):
                mu_exp = mu.expand(-1, -1, -1, u.shape[-2], u.shape[-1])  # (B,C,C,H,W)
                sigma_exp = sigma.expand(-1, -1, -1, u.shape[-2], u.shape[-1])
                return 2 * torch.exp(-((u - mu_exp) ** 2 / (sigma_exp) ** 2) / 2) - 1

            return growth

    @torch.no_grad()
    def step(self):
        """
        Performs one step of the Lenia update.
        """

        U = self.kernel_fftconv(self.state)  # (B,C*k_mult,C,H,W)

        assert (self.h, self.w) == (U.shape[-2], U.shape[-1])

        weights = self.weights[..., None, None]  # (B,C*k_mult,C,1,1)
        weights = weights.expand(-1, -1, -1, self.h, self.w)  # (B,C*k_mult,C,H,W)

        # Weight normalized growth :
        dx = (self.growth(U) * weights).sum(
            dim=1
        )  # (B,C*k_mult,H,W) # G(U)[:,i,j] is contribution of channel i to channel j

        # Apply growth and clamp
        self.state = torch.clamp(self.state + self.dt * dx, 0, 1)  # (B,C,H,W)

        self.frames += 1  # Increment frame count

    def kernel_fftconv(self, state):
        """
        Compute convolution with the state using the fft kernel.

        Args:
            state : (B,C*k_mult,C,H,W), state of the automaton
        """
        state = torch.fft.fft2(state)  # (B,C,H,W) fourier transform
        state = state[:, :, None, None, :]  # (B,C,1,1,H,W)
        fft_unfolded = self.fft_kernel.reshape(
            self.batch, self.C, self.k_mult, self.C, self.h, self.w
        )  # (B,C*k_mult,C,h,w)
        state = state * fft_unfolded  # (B,C,k_mult,C,H,W), convoluted
        state = state.reshape(self.batch, self.C * self.k_mult, self.C, self.h, self.w)  # (B,C*k_mult,C,H,W)
        state = torch.fft.ifft2(state)  # (B,C*k_mult,C,H,W), back to spatial domain

        return torch.real(state)

    def mass(self):
        """
        Computes average 'mass' of the automaton for each channel

        returns :
        mass : (B,C) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1, -2))  # (B,C) mean mass for each color

    @torch.no_grad()
    def draw(self):
        """
        Draws the RGB worldmap from state. Should be called before dislaying the worldmap/worldsurface.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"

        toshow = self.state[0].clone()  # (C,H,W), pygame conversion done later

        if self.C == 1:
            toshow = toshow.expand(3, -1, -1)  # (3,H,W)
        elif self.C == 2:
            toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
        else:
            toshow = toshow[:3, :, :]  # (3,H,W)

        if self.display_kernel:
            toshow = self._draw_kernel(toshow)

        self._worldmap = torch.clamp(toshow, 0.0, 1.0)

    def _draw_kernel(self, image):
        """
        Draws the kernel on the image.
        """

        horizontal_fit = self.w // self.k_size  # number of kernels that fit in the width
        kern = self.compute_ker()  # (C*k_mult,3,k_size,k_size)
        for i in range(kern.shape[0]):
            height_offset = i // horizontal_fit + 1
            image[
                :,
                self.h - height_offset * self.k_size : self.h - (height_offset - 1) * self.k_size,
                (i % horizontal_fit) * self.k_size : (i % horizontal_fit + 1) * self.k_size,
            ] = kern[i].cpu()
        return image  # (3,H,W)

    def _random_params(self, true_random=False):
        """
        Generates random parameters for the Lenia automaton.
        If true_random is True, uses the truerandom generation method.
        """
        if true_random:
            params = LeniaParams.random_gen(
                batch_size=self.batch,
                num_channels=self.C,
                device=self.device,
                k_size=self.k_size,
                k_mult=self.k_mult,
            )
        else:
            params = LeniaParams.default_gen(
                batch_size=self.batch,
                num_channels=self.C,
                device=self.device,
                k_size=self.k_size,
                k_mult=self.k_mult,
            )

        self.update_params(params, k_size_override=None)  # Update the parameters
        self.set_init_circle()  # Reset the state to circle

    def reroll_params(self, true_random=False):
        """
        Rerolls the parameters of the automaton, keeping the same batch size and number of channels.
        If true_random is True, uses the truerandom generation method.
        """
        if true_random:
            params = LeniaParams.random_gen(
                batch_size=self.batch,
                num_channels=self.C,
                device=self.device,
                k_size=self.k_size,
                k_mult=self.k_mult,
            )
        else:
            params = LeniaParams.default_gen(
                batch_size=self.batch,
                num_channels=self.C,
                device=self.device,
                k_size=self.k_size,
                k_mult=self.k_mult,
            )
        self.update_params(params)
        
    def process_event(self, event, camera=None):  # This method is used to process the pygame events
        """
        N (+ shift) -> New random (truerandom) parameters
        A -> Random params, with arbitrary function (if active)
        W (+shift) -> Reroll kernel (growth)
        M -> Load new interesting param
        U -> Variate around parameters
        I (+shift/+ctrl) -> Intialize with perlin noise
        O -> Initialize with circle
        S (+shift) -> Save the current parameters (+state)
        K -> Toggle display kernel
        Y (+shift) -> Toggle arbi random param generation
        DEL -> sets state to 0
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.reroll_params(true_random=True)
                else:
                    self.reroll_params(true_random=False)
            if event.key == pygame.K_a:
                params = LeniaParams.mixed_gen(
                    batch_size=self.batch,
                    num_channels=self.C,
                    device=self.device,
                    k_size=self.k_size,
                    k_mult=self.k_mult,
                    k_rescale=(-0.5, 1),
                    k_arbi=self.k_arbi,
                    g_arbi=self.g_arbi,
                    sigma_size=0.8,
                    k_coeffs=4,
                    g_coeffs=3,
                    g_clip=-0.5,
                )
                self.update_params(params, k_size_override=None)

            if event.key == pygame.K_u:
                """ Variate around parameters"""
                mutated_params = self.params.mutate(magnitude=0.1, rate=0.8)
                self.update_params(mutated_params, k_size_override=None)
            if event.key == pygame.K_i:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Intialize with fractal perlin
                    self.set_init_fractal()
                elif pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Initialize with random wavelength perlin
                    sq_size = random.randint(5, min(self.h, self.w))
                    self.set_init_perlin(wavelength=sq_size)
                else:
                    self.set_init_perlin()
            if event.key == pygame.K_o:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.set_init_circle(radius=5 * self.k_size)
                else:
                    # Initialize with circle
                    self.set_init_circle(fractal=False)

            if event.key == pygame.K_w:
                # Reroll growth/kernel
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.params.reroll_params(kernel=False, arbi=self.g_arbi)
                else:
                    self.params.reroll_params(kernel=True, arbi=self.k_arbi)
                    # print('Rerolled noob, new C : ', self.params.num_channels)
                self.update_params(self.params, k_size_override=None)  # Translate to arbi if needed

            if event.key == pygame.K_y:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.g_arbi = not self.g_arbi
                else:
                    self.k_arbi = not self.k_arbi

                self.update_params(self.params, k_size_override=None)  # Translate to arbi if needed
            if event.key == pygame.K_m:
                self.load_interest_file()
            if event.key == pygame.K_s:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Save current state + parameters
                    self._save_with_state(self.save_dir)
                else:
                    # Save the current parameters to remarkable dir :
                    self._save(self.save_dir)
            if event.key == pygame.K_k:
                # Toggle display kernel
                self.display_kernel = not self.display_kernel
            if event.key == pygame.K_DELETE | pygame.K_BACKSPACE:
                self.frames = 0
                self.state = torch.zeros_like(self.state)
        
        mouse_state = self.get_mouse_state(camera)
        if (mouse_state.left or mouse_state.right) and mouse_state.inside:
            
            x, y = mouse_state.x, mouse_state.y
            
            if mouse_state.left:
                self._affect_matter(x,y, add = self.toggle_add_remove.state)
            elif mouse_state.right:
                self._affect_matter(x,y, add = not self.toggle_add_remove.state)

        for component in self.changed_components:
            if component == self.reset_button:
                self.set_init_circle()
            if component == self.cool_rule:
                self.load_interest_file()
                self.set_init_circle()
            if component == self.random_rule:
                self.reroll_params(true_random=not self.toggle_rand_type.state)
                self.set_init_circle()
            if component == self.wipe_button:
                self.frames = 0
                self.state = torch.zeros_like(self.state)
    
    def _affect_matter(self, x, y, add=True):
        add_rad = self.k_size / 2.0
        add_mask = (self.X - x) ** 2 + (self.Y - y) ** 2 < add_rad**2  # (H,W)

        if add:
            addition = torch.rand((self.batch, self.C, self.h, self.w), device=self.device)
            self.state[:, :, add_mask] += self.add_speed *0.05 * addition[:, :, add_mask]
        else:
            self.state[:, :, add_mask] -= 0.05*self.add_speed
            self.state[:, :, add_mask] = torch.clamp(self.state[:, :, add_mask], 0.0, 1.0)

    def load_interest_file(self):
        if self.interest_files:
            # Load random interesting param, if we have some
            file = self.interest_files[self.chosen_interesting]  # To add as parameter
            self.chosen_interesting = (self.chosen_interesting + 1) % len(self.interest_files)

            params = LeniaParams(from_file=file, device=self.device)
            self.update_params(params, k_size_override=None)
            print("Loaded : ", file)

    def compute_ker(self, batch=0):
        """
        Prepares the kernel and translate it to an RGB image for viewing.

        returns :
        kern : (C,3,k_size,k_size) tensor, kernel as
        """
        kern = self.k[batch].detach()  # (C,C, k_size, k_size), removed batch

        if kern.shape[1] == 1:
            kern = kern.expand(-1, 3, -1, -1)
        elif kern.shape[1] > 3:
            kern = kern[:3, :3]  # Cut, and include only the first 3 set of kernels

        maxs = torch.tensor([torch.max(kern[i]) for i in range(kern.shape[0])], device=self.device)  # (C,)
        # print(maxs)
        maxs = maxs[:, None, None, None]
        kern = kern / maxs

        return kern  # (C,3,k_size,k_size)

    def _save_with_state(self, path):
        """
        Saves the parameters of the automaton ALONG with the current state.

        Args:
            path : str, folder to save the parameters to
        """
        path = Path(path) / "state_saves"
        path.mkdir(parents=True, exist_ok=True)

        to_save = deepcopy(self.params)
        to_save.state = self.state
        to_save.save_indiv(path, batch_name=True, annotation=["_state"])

    def _save(self, path):
        """
        Saves the parameters of the automaton to the given path.

        Args:
            path : str, folder to save the parameters to
        """
        to_save = deepcopy(self.params)
        to_save["g_arbi"] = self.g_arbi
        to_save["k_arbi"] = self.k_arbi

        to_save.save_indiv(path, batch_name=False)

    def _load_state(self, state):
        """
        Loads a state into the automaton, without breaking
        if the provided state has a different shape than the automaton
        """
        B, C, H, W = state.shape
        if not (B == self.batch or B == 1):
            print("Skipping, batch size of state should match automaton or be 1")
            return
        if not (C == self.C):
            print("Skipping, number of channels of state should match automaton")
            return
        if H > self.h or W > self.w:
            print("Warning, state will be clipped to automaton size")
            if H > self.h:
                clip_h = (H - self.h) // 2
                extra_clip = 0 if (H - self.h) % 2 == 0 else 1
                state = state[:, :, clip_h : H - clip_h - extra_clip, :]  # Symmetrical clip
                H = self.h
            if W > self.w:
                clip_w = (W - self.w) // 2
                extra_clip = 0 if (W - self.w) % 2 == 0 else 1
                state = state[:, :, :, clip_w : W - clip_w - extra_clip]  # Symmetrical clip
                W = self.w

        self.state = state[:, :, : self.h, : self.w]

        if H < self.h:
            pad_h = (self.h - H) // 2
        else:
            pad_h = 0
        if W < self.w:
            pad_w = (self.w - W) // 2
        else:
            pad_w = 0

        self.state = F.pad(self.state, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

    def get_string_state(self):
        return f"g_arb : {self.g_arbi}, k_arb : {self.k_arbi}, {self.frames} F"


