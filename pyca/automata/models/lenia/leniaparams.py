import torch
import math
from .funcgen import ArbitraryFunction
from ...utils.batch_params import BatchParams

class LeniaParams(BatchParams):
    """
    Class handling Lenia parameters

    Keys:
        'k_size' : odd int, size of kernel used for computations
        'mu' : (B,C*k_mult,C) tensor, mean of growth functions
        'sigma' : (B,C*k_mult,C) tensor, standard deviation of the growth functions
        'beta' :  (B,C*k_mult,C, # of rings) float, max of the kernel rings
        'mu_k' : (B,C*k_mult,C, # of rings) [0,1.], location of the kernel rings
        'sigma_k' : (B,C*k_mult,C, # of rings) float, standard deviation of the kernel rings
        'weights' : (B,C*k_mult,C) float, weights for the growth weighted sum
        'k_harmonics' : (B,C*k_mult,C, # of harmonics) floats, harmonics frequencies for the kernel (optional)
        'k_coeffs' : (B,C*k_mult,C, # of harmonics) floats, harmonics coefficients for the kernel (optional)
        'k_rescale' : tuple of floats, rescale range for the kernel (optional)
        'k_clip' : float, clip value for the kernel (optional)
        'g_harmonics' : (B,C*k_mult,C, # of harmonics) floats, harmonics frequencies for the growth (optional)
        'g_coeffs' : (B,C*k_mult,C, # of harmonics) floats, harmonics coefficients for the growth (optional)
        'g_rescale' : tuple of floats, rescale range for the growth (optional)
        'g_clip' : float, clip value for the growth (optional)
    """

    def __init__(
        self,
        param_dict=None,
        from_file=None,
        k_size=None,
        batch_size=None,
        channels=3,
        k_mult=1,
        device="cpu",
    ):
        """
        Args:
            from_file : str, path to file containing parameters. Priority over param_dict
            param_dict : dict, dictionary of parameters
            k_size : int, size of the kernel. Used if both param_dict and from_file are None
            batch_size : int, number of parameters in the batch. Used if both param_dict and from_file are None
            channels : int, number of channels in the automaton
            device : str, device to use
        """
        if param_dict is None and from_file is None:
            assert k_size is not None and batch_size is not None, (
                "k_size and batch_size must be provided if no parameters are given"
            )

            param_dict = LeniaParams.default_gen(
                batch_size=batch_size, num_channels=channels, k_size=k_size, k_mult=k_mult, device=device
            ).param_dict  # dis very ugly but not sure how to do it better
            super().__init__(param_dict=param_dict, device=device)
        else:
            super().__init__(param_dict=param_dict, from_file=from_file, batch_size=batch_size, device=device)

        assert "weights" in self.param_dict.keys(), 'LeniaParams need "weights" tensor'
        assert "k_size" in self.param_dict.keys(), 'LeniaParams need "k_size" value'
        if "k_mult" not in self.param_dict.keys():
            self.k_mult = 1  # legacy param that has no k_mult
        if "num_channels" not in self.param_dict.keys():
            self.num_channels = channels  # legacy param that has no num_channels
        self._sanitize()
        self.to(device)

    def _sanitize(self):
        """
        Sanitizes the parameters by clamping them to valid values.
        """
        param_keys = self.param_dict.keys()
        if "mu" in param_keys:
            self.mu = torch.clamp(self.mu, -2, 2)
        if "sigma" in param_keys:
            self.sigma = torch.clamp(self.sigma, 1e-4, 1.0)
        if "beta" in param_keys:
            self.beta = torch.clamp(self.beta, 0, None)
        if "mu_k" in param_keys:
            self.mu_k = torch.clamp(self.mu_k, 0.0, 2.0)
        if "sigma_k" in param_keys:
            self.sigma_k = torch.clamp(self.sigma_k, 1e-4, 1.0)

        self.weights = torch.clamp(self.weights, 0, None)
        ## Normalize weights
        N = self.weights.sum(dim=1, keepdim=True)  # (B,1,C)
        self.weights = torch.where(N > 1.0e-6, self.weights / N, 0)

    def reroll_params(self, kernel=True, arbi=False):
        """
        Rerolls part of the parameters (growth or kernels).

        Args:
            kernels : if True, rerolls the kernels, otherwise rerolls the growth functions
            arbi : if True, rerolls with arbitrary functions, otherwise rerolls with random_gen
        """
        if kernel:
            if arbi:
                from_fourier = LeniaParams._fourier_master(
                    batch_size=self.batch_size,
                    k_mult=self.k_mult,
                    num_channels=self.param_dict["num_channels"],
                    decay=0.3,
                    harmonics=(1.0, 4),
                    ranges=(0.0, 1.0),
                    rescale=(-0.7, 1.0),
                    clip=0.0,
                    device=self.device,
                )
                self.k_coeffs = from_fourier["coeffs"]
                self.k_harmonics = from_fourier["harmonics"]
                self.k_rescale = from_fourier["rescale"]
                self.k_clip = from_fourier["clip"]
            else:
                from_random_gen = LeniaParams.random_gen(
                    batch_size=self.batch_size,
                    num_channels=self.num_channels,
                    k_mult=self.k_mult,
                    k_size=self.k_size,
                    device=self.device,
                ).param_dict
                self.mu_k = from_random_gen["mu_k"]
                self.sigma_k = from_random_gen["sigma_k"]
                self.beta = from_random_gen["beta"]
        else:
            if arbi:
                from_fourier = LeniaParams._fourier_master(
                    batch_size=self.batch_size,
                    num_channels=self.param_dict["num_channels"],
                    k_mult=self.k_mult,
                    decay=0.2,
                    harmonics=(1.0, 3),
                    ranges=(0.0, 2.0),
                    rescale=(-1.0, 1.0),
                    clip=-0.5,
                    device=self.device,
                )
                self.g_coeffs = from_fourier["coeffs"]
                self.g_harmonics = from_fourier["harmonics"]
                self.g_rescale = from_fourier["rescale"]
                self.g_clip = from_fourier["clip"]
            else:
                from_random_gen = LeniaParams.random_gen(
                    batch_size=self.batch_size,
                    num_channels=self.num_channels,
                    k_mult=self.k_mult,
                    k_size=self.k_size,
                    device=self.device,
                ).param_dict
                self.mu = from_random_gen["mu"]
                self.sigma = from_random_gen["sigma"]

        self._sanitize()

    @staticmethod
    def mixed_gen(
        batch_size,
        num_channels=3,
        sigma_size=1.0,
        k_size=None,
        k_mult=1,
        k_arbi=True,
        g_arbi=False,
        k_coeffs=3,
        k_rescale=(-0.3, 1.0),
        g_coeffs=4,
        g_rescale=(-1.0, 1.0),
        g_clip=None,
        device="cpu",
    ):
        """
        Mixed generation between arbitrary and standard Lenia parameters.
        With k_arbi and g_arbi set to False, it is the same as random_gen.
        """
        params = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        from_random_gen = LeniaParams.random_gen(
            batch_size=batch_size,
            num_channels=num_channels,
            sigma_size=sigma_size,
            k_size=k_size,
            k_mult=k_mult,
            device=device,
        ).param_dict
        # from_random_gen = LeniaParams.default_gen(batch_size=batch_size,num_channels=num_channels,k_size=k_size,device=device).param_dict
        if k_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                k_mult=k_mult,
                decay=None,
                harmonics=(0.0, k_coeffs),
                ranges=(0.0, 1.0),
                rescale=k_rescale,
                clip=0.0,
                device=device,
            )
            params["k_coeffs"] = from_fourier["coeffs"]
            params["k_harmonics"] = from_fourier["harmonics"]
            params["k_rescale"] = from_fourier["rescale"]
            params["k_clip"] = from_fourier["clip"]
        else:
            params["mu_k"] = from_random_gen["mu_k"]
            params["sigma_k"] = from_random_gen["sigma_k"]
            params["beta"] = from_random_gen["beta"]

        if g_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                decay=None,
                harmonics=(0.0, g_coeffs),
                ranges=(0.0, 2.0),
                k_mult=k_mult,
                clip=g_clip,
                rescale=g_rescale,
                device=device,
            )
            params["g_coeffs"] = from_fourier["coeffs"]
            params["g_harmonics"] = from_fourier["harmonics"]
            params["g_rescale"] = from_fourier["rescale"]
            params["g_clip"] = from_fourier["clip"]
        else:
            params["mu"] = from_random_gen["mu"]
            params["sigma"] = from_random_gen["sigma"]

        return LeniaParams(params, device=device)

    @staticmethod
    def exp_decay_gen(
        batch_size,
        num_channels=3,
        k_arbi=True,
        g_arbi=False,
        k_size=None,
        k_mult=1,
        k_decay=1.0,
        k_harmo_start=1.0,
        g_decay=1.0,
        g_harmo_start=1.0,
        k_coeffs=6,
        k_rescale=(-0.3, 1.0),
        g_coeffs=4,
        g_rescale=(-2.0, 2.0),
        g_clip=None,
        device="cpu",
    ):
        """
        Generates parameters with exponential decay.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        params = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        from_random_gen = LeniaParams.random_gen(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        ).param_dict

        if k_arbi:
            from_exp_gen = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                decay=k_decay,
                k_mult=k_mult,
                harmonics=(k_harmo_start, k_coeffs),
                ranges=(0.0, 1.0),
                rescale=k_rescale,
                clip=0.0,
                device=device,
            )

            for key, value in from_exp_gen.items():
                params["k_" + key] = value
        else:
            for key in ["mu_k", "sigma_k", "beta"]:
                params[key] = from_random_gen[key]

        if g_arbi:
            from_exp_gen = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                decay=g_decay,
                k_mult=k_mult,
                harmonics=(g_harmo_start, g_coeffs),
                ranges=(0.0, 2.0),
                rescale=g_rescale,
                clip=g_clip,
                device=device,
            )

            for key, value in from_exp_gen.items():
                params["g_" + key] = value
        else:
            params["mu"] = from_random_gen["mu"]
            params["sigma"] = from_random_gen["sigma"]

        return LeniaParams(params, device=device)

    @staticmethod
    def default_gen(batch_size, num_channels=3, k_size=None, k_mult=1, device="cpu"):
        """
        Generates (standard) parameters with random values. Prior tuned to
        be close to 'stability' in Lenia.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """

        params_base = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        mu = 0.7 * torch.rand((batch_size, num_channels * k_mult, num_channels), device=device)
        sigma = (
            mu
            / (math.sqrt(2 * math.log(2)))
            * 0.8
            * torch.rand((batch_size, num_channels * k_mult, num_channels), device=device)
            + 1e-4
        )

        params = {
            "mu": mu,
            "sigma": sigma,
            "beta": torch.rand((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "mu_k": 0.5
            + 0.2 * torch.randn((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "sigma_k": 0.05
            * (
                1
                + torch.clamp(
                    0.3 * torch.randn((batch_size, num_channels * k_mult, num_channels, 3), device=device),
                    min=-0.9,
                )
                + 1e-4
            ),
        }

        params.update(params_base)
        return LeniaParams(params, device=device)

    @staticmethod
    def random_gen(batch_size, num_channels=3, k_size=None, k_mult=1, sigma_size=1.0, device="cpu"):
        """
        Full random generation for standard Lenia Parameters. Weights are biased towards
        self-interaction.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        params_base = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )
        mu = torch.rand((batch_size, num_channels * k_mult, num_channels), device=device)
        sigma = (
            sigma_size * torch.rand((batch_size, num_channels * k_mult, num_channels), device=device) + 1e-4
        )

        params = {
            "mu": mu,
            "sigma": sigma,
            "beta": torch.rand((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "mu_k": torch.rand((batch_size, num_channels * k_mult, num_channels, 3), device=device),
            "sigma_k": 0.05
            * (
                1
                + torch.clamp(
                    0.3 * torch.randn((batch_size, num_channels * k_mult, num_channels, 3), device=device),
                    min=-0.9,
                )
                + 1e-4
            ),
        }
        params.update(params_base)

        return LeniaParams(params, device=device)

    @staticmethod
    def fourier_range_gen(
        batch_size,
        num_channels=3,
        k_size=None,
        k_mult=1,
        k_arbi=True,
        g_arbi=False,
        k_harmonics: torch.Tensor = torch.tensor([2, 3, 4]),
        k_rescale=(-0.3, 1.0),
        g_harmonics: torch.Tensor = torch.tensor([1, 2]),
        g_rescale=(-1.0, 1.0),
        g_clip=None,
        device="cpu",
    ):
        """
        Generates parameters with Fourier range.

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        params = LeniaParams._universal_params(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        )

        from_random_gen = LeniaParams.random_gen(
            batch_size=batch_size, num_channels=num_channels, k_size=k_size, k_mult=k_mult, device=device
        ).param_dict

        if k_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                harmonics=k_harmonics,
                k_mult=k_mult,
                ranges=(0.0, 1.0),
                rescale=k_rescale,
                device=device,
            )
            params["k_coeffs"] = from_fourier["coeffs"]
            params["k_harmonics"] = from_fourier["harmonics"]
            params["k_rescale"] = from_fourier["rescale"]
            params["k_clip"] = from_fourier["clip"]
        else:
            params["mu_k"] = from_random_gen["mu_k"]
            params["sigma_k"] = from_random_gen["sigma_k"]
            params["beta"] = from_random_gen["beta"]

        if g_arbi:
            from_fourier = LeniaParams._fourier_master(
                batch_size=batch_size,
                num_channels=num_channels,
                harmonics=g_harmonics,
                k_mult=k_mult,
                ranges=(0.0, 2.0),
                rescale=g_rescale,
                clip=g_clip,
                device=device,
            )
            params["g_coeffs"] = from_fourier["coeffs"]
            params["g_harmonics"] = from_fourier["harmonics"]
            params["g_rescale"] = from_fourier["rescale"]
            params["g_clip"] = from_fourier["clip"]
        else:
            params["mu"] = from_random_gen["mu"]
            params["sigma"] = from_random_gen["sigma"]

        return LeniaParams(params, device=device)

    @staticmethod
    def to_arbi_params(
        lenia_params: "LeniaParams", discretization_points=1000, device="cpu"
    ) -> "LeniaParams":
        """
        Convert standard Lenia parameters to arbitrary function parameters.

        Args:
            lenia_params : LeniaParams, parameters to convert
            discretization_points : int, number of points to use for the discretization of the functions
            device : str, device to use
        """

        def k_func(mu_k, sigma_k, beta):
            x_range = torch.arange(
                0.0, 1, step=1 / discretization_points, device=device
            )  # (discretization_points,)
            x_range = x_range[None, None, None, None, :]  # (1,1,1,1,discretization_points)

            K = torch.exp(
                -(((x_range - mu_k[..., None]) / sigma_k[..., None]) ** 2) / 2.0
            )  # (B,C,C,#of rings, discretization_points)
            beta = beta[..., None]  # (B,C,C,#of rings, 1)
            K = torch.sum(beta * K, dim=-2)  # (B,C,C,discretization_points)

            return K

        def g_func(mu, sigma):
            x_range = torch.arange(
                0, 2, step=2 / discretization_points, device=device
            )  # (discretization_points,)
            x_range = x_range[None, None, None, :]  # (1,1,1,discretization_points)

            G = (
                2 * torch.exp(-(((x_range - mu[..., None]) / sigma[..., None]) ** 2) / 2.0) - 1
            )  # (B,C,C,discretization_points)
            return G

        k_mult = lenia_params.param_dict.get("k_mult", 1)
        if "mu" in lenia_params:
            B, _, C = lenia_params.mu.shape
            g_evals = g_func(lenia_params.mu, lenia_params.sigma)  # (B,C,C,discretization_points)
            g_evals = g_evals.reshape(B * C * k_mult * C, discretization_points)
            g_coeffs = 8
            g_arbi = ArbitraryFunction.from_function_evals(
                g_evals, (0.0, 2.0), n_coeffs=g_coeffs, device=device
            )
            g_coeffs_val = g_arbi.coefficients.reshape(B, C * k_mult, C, 2 * g_coeffs)
            g_harmonics = g_arbi.harmonics.reshape(B, C * k_mult, C, g_coeffs)
        else:
            g_coeffs_val = lenia_params.g_coeffs
            g_harmonics = lenia_params.g_harmonics

        if "mu_k" in lenia_params:
            B, _, C, _ = lenia_params.mu_k.shape
            k_evals = k_func(
                lenia_params.mu_k, lenia_params.sigma_k, lenia_params.beta
            )  # (B,C,C,discretization_points)
            k_evals = k_evals.reshape(B * C * k_mult * C, discretization_points)
            k_coeffs = 15
            k_arbi = ArbitraryFunction.from_function_evals(
                k_evals, (0.0, 1.0), n_coeffs=k_coeffs, device=device
            )
            k_coeffs_val = k_arbi.coefficients.reshape(B, C * k_mult, C, 2 * k_coeffs)
            k_harmonics = k_arbi.harmonics.reshape(B, C * k_mult, C, k_coeffs)
        else:
            k_coeffs_val = lenia_params.k_coeffs
            k_harmonics = lenia_params.k_harmonics

        params = {
            "k_size": lenia_params.k_size,
            "k_mult": k_mult,
            "k_coeffs": k_coeffs_val,
            "k_harmonics": k_harmonics,
            "g_coeffs": g_coeffs_val,
            "g_harmonics": g_harmonics,
            "weights": lenia_params.weights,
        }

        return LeniaParams(params, device=device)

    # ========== BELOW, UTILITY FUNCTIONS FOR GENERATING ARBI PARAMETERS ==========
    @staticmethod
    def _fourier_master(
        batch_size,
        num_channels,
        harmonics,
        ranges,
        k_mult=1,
        decay=None,
        rescale=None,
        clip=None,
        device="cpu",
    ):
        """
        Master fourier function generator. Has all the options for generating Fourier functions parameters.

        Args:
            batch_size : number of parameters to generate
            num_channels : number of channels in the automaton
            harmonics : determines the harmonics to be used. Has many modes listed below:
                - tuple (harmo_start, num_harmonics) : will generate num_harmonics harmonics starting from harmo_start
                - float tensor (num_harmonics,) : will use the harmonics in the specified tensor, for all batches and channels
                - float tensor (batch_size*num_channels*num_channels,num_harmonics) : will use the harmonics in the specified tensor, can be different for each batch and channel
            (num_harmonics,) integers or (batch_size*num_channels*num_channels,num_harmonics) harmonics to be used
            decay : float, decay rate for the exponential function
            ranges : tuple, x-range for the functions. NOTE : could support different ranges for the functions, but not useful right now.
            rescale : tuple, (min,max) range of the x values for the function to be rescaled to
            clip : float, after rescaling will clip to this value
            device : device on which to generate the parameters
        """
        if isinstance(harmonics, tuple):
            harmo_start, num_harmonics = harmonics
            harmonics = torch.arange(harmo_start, harmo_start + num_harmonics, device=device)[None, :].expand(
                batch_size * num_channels * k_mult * num_channels, num_harmonics
            )
        elif isinstance(harmonics, torch.Tensor):
            if len(harmonics.shape) == 1:
                harmonics = harmonics[None, :].expand(
                    batch_size * num_channels * k_mult * num_channels, num_harmonics
                )
            elif len(harmonics.shape) == 2:
                assert harmonics.shape[0] == batch_size * num_channels * num_channels * k_mult, (
                    f"Expected harmonics tensor of shape ({batch_size * num_channels * num_channels * k_mult},{harmonics.shape[1]}), got {harmonics.shape}"
                )

        num_harmonics = harmonics.shape[1]
        coeffs = torch.randn(
            (batch_size, num_channels * k_mult, num_channels, num_harmonics, 2), device=device
        )  # (batch_size,num_channels,num_channels,num_harmonics,2)
        if decay is not None:
            dampening = torch.exp(-decay * torch.arange(0, num_harmonics, device=device).float())[
                None, None, None, :, None
            ]
            coeffs = coeffs * dampening

        coeffs = coeffs.reshape(
            (batch_size, num_channels * k_mult, num_channels, num_harmonics * 2)
        )  # (batch_size,num_channels,num_channels,num_harmonics*2)
        harmonics = harmonics.reshape(
            batch_size, num_channels * k_mult, num_channels, num_harmonics
        )  # (batch_size,num_channels,num_channels,num_harmonics)
        return {
            "coeffs": coeffs,
            "harmonics": harmonics,
            "rescale": rescale,
            "clip": clip,
            "ranges": ranges,
        }

    @staticmethod
    def _universal_params(batch_size, num_channels, k_size=31, k_mult=1, diag_inhibition=0.8, device="cpu"):
        """
        Return bare minimum k_size and weights dictionary for the model.

        Args:
            batch_size : number of parameters to generate
            num_channels : number of channels in the automaton
            k_size : int, size of the kernel
            k_mult : int, multiplier for the number of source channels
            diag_inhibition : float, value of the diagonal inhibition
            device : device on which to generate the parameters
        """
        if k_mult == 1:
            return {
                "k_size": k_size,
                "weights": torch.rand(batch_size, num_channels, num_channels, device=device)
                * (1 - diag_inhibition * torch.diag(torch.ones(num_channels, device=device))),
                "k_mult": k_mult,
                "num_channels": num_channels,
            }
        else:
            # No diag inhibition possible for k_mult > 1
            return {
                "k_size": k_size,
                "weights": torch.rand(batch_size, num_channels * k_mult, num_channels, device=device),
                "k_mult": k_mult,
                "num_channels": num_channels,
            }


if __name__ == "__main__":
    test = LeniaParams(batch_size=1, k_size=21)
