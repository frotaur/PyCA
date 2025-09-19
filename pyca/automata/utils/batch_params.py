import torch
import math, os
from .hash_params import params_to_words


class BatchParams:
    """
    Class handling general batched parameters.
    It has a dictionary of parameters which are torch tensors (or int) of various
    sizes, with a batch dimension. Specific keys are model specific.
    """

    def __init__(self, param_dict=None, from_file=None, batch_size=None, device="cpu"):
        """
        Args:
            param_dict : dict, dictionary of parameters
            from_file : str, path to file containing parameters. Priority over param_dict
            batch_size : int, number of parameters in the batch. Used if both param_dict and from_file are None
            device : str, device to use
        """
        self.device = device
        self.batch_size = batch_size

        if param_dict is None and from_file is None:
            assert self.batch_size is not None, "batch_size must be provided if no parameters are given"
            param_dict = {}
            print("Warning: initialized empty BatchParams")
        elif from_file is not None:
            param_dict = torch.load(from_file, map_location=device, weights_only=True)

        self.param_dict = {}
        for key in param_dict.keys():
            if isinstance(param_dict[key], torch.Tensor):
                if self.batch_size is None:
                    self.batch_size = param_dict[key].shape[0]
                else:
                    assert self.batch_size == param_dict[key].shape[0], (
                        "All tensors in param_dict must have the same batch size"
                    )
            self.__setattr__(key, param_dict[key])
        if self.batch_size is None:
            print("Couldn't infer batch size from parameters, setting to 1")
            self.batch_size = 1

        self.to(device)

    def __setattr__(self, name, value):
        if not name in {"param_dict", "batch_size", "device"}:
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == self.batch_size, (
                    f"Attempted to add element of incorrect batch size: got {value.shape[0]} expected {self.batch_size}"
                )
            self.param_dict[name] = value

        super().__setattr__(name, value)

    @property
    def name(self):
        """
        Returns a string representation of the parameters.
        """
        return params_to_words(self.param_dict)

    def to(self, device):
        """
        Moves the parameters to a device, like pytorch.
        """
        self.device = device
        for key in self.param_dict.keys():
            if isinstance(self.param_dict[key], torch.Tensor):
                self.param_dict[key] = self.param_dict[key].to(device)

    def save_indiv(self, folder, batch_name=False, annotation=None):
        """
        Saves parameter individually.

        Args:
        folder : path to folder where to save params individually
        params : dictionary of parameters
        batch_name : if True, names indiv parameters with batch name + annotation
        annotation : list of same length as batch_size, an annotation of the parameters.
            Only used if batch_name is True
        """
        os.makedirs(folder, exist_ok=True)

        name = params_to_words(self.param_dict)
        batch_size = self.batch_size

        params_list = [self[i] for i in range(batch_size)]

        if annotation is None:
            annotation = [f"{j:02d}" for j in range(len(params_list))]
        else:
            assert len(annotation) == len(params_list), (
                f"Annotation (len={len(annotation)}) must \
            have same length as batch_size (len={len(params_list)})"
            )

        for j in range(len(params_list)):
            if not batch_name:
                indiv_name = params_to_words(params_list[j].param_dict)
                fullname = indiv_name + ".pt"
            else:
                fullname = name + f"_{annotation[j]}" + ".pt"

            torch.save(params_list[j].param_dict, os.path.join(folder, fullname))

    def save(self, folder, name=None):
        """
        Saves the (batched) parameters to a folder.
        """
        os.makedirs(folder, exist_ok=True)
        if name is None:
            name = params_to_words(self.param_dict)

        torch.save(self.param_dict, os.path.join(folder, name + ".pt"))

    def load(self, path):
        """
        Loads the parameters from a file.
        """
        self.__init__(from_file=path)

    def __mul__(self, scalar: float) -> "BatchParams":
        """
        Multiplies all parameters by a scalar.
        """
        if not isinstance(scalar, (int, float)):
            raise ValueError("Can only multiply by a scalar")
        new_params = {}
        for key in self.param_dict.keys():
            if not isinstance(self.param_dict[key], torch.Tensor):
                # Assume non-tensor parameters are not to be multiplied
                new_params[key] = self.param_dict[key]
            else:
                new_params[key] = self.param_dict[key] * scalar

        return type(self)(param_dict=new_params, device=self.device)

    def __rmul__(self, scalar: float) -> "BatchParams":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "BatchParams":
        return self.__mul__(1.0 / scalar)

    def __contains__(self, key):
        return key in self.param_dict.keys()

    def __getitem__(self, idx):
        """
        Works as a dictionary, indexing the parameters with strings, or
        as advanced indexing on the batch dimension, like in pytorch.
        Will ALWAYS keep at least one dimension for the batch size.
        In other words,params[1] is the same as params[1:2]

        WARNING : will not fail if the key is not found, will return None.
        """
        if isinstance(idx, str):
            return self.param_dict.get(idx, None)  # Soft fail if key not found, return None.
        elif isinstance(idx, int):
            idx = slice(idx, idx + 1)

        params = {}
        for k, v in self.param_dict.items():
            if not isinstance(v, torch.Tensor):
                params[k] = v
            else:
                params[k] = v[idx]

        return type(self)(params, device=self.device)

    def __setitem__(self, idx, value):
        """
        Works as a dictionary, setting the parameters with strings, or
        as advanced indexing on the batch dimension, like in pytorch.
        Will ALWAYS keep at least one dimension for the batch size.
        In other words,params[1] is the same as params[1:2]
        """
        if isinstance(idx, str):
            self.__setattr__(idx, value)
        else:
            # if idx is not a string, we assume value is BatchParams
            assert isinstance(value, BatchParams), (
                "Can only setitem with BatchParams when using advanced indexing"
            )
            assert value.param_dict.keys() == self.param_dict.keys(), (
                "Keys of the two BatchParams do not match"
            )
            if isinstance(idx, int):
                idx = slice(idx, idx + 1)

            # Else, assume its some advanced pytorch indexing
            for k, v in value.param_dict.items():
                if not isinstance(v, torch.Tensor):
                    self.__setattr__(k, v)
                else:
                    self.param_dict[k][idx] = v

    def __add__(self, other: "BatchParams") -> "BatchParams":
        """
        Adds two sets of parameters together.
        """
        assert self.batch_size == other.batch_size, "Batch sizes do not match"

        new_params = {}
        for key in self.param_dict.keys():
            if not isinstance(self.param_dict[key], torch.Tensor):
                assert self.param_dict[key] == other.param_dict[key], (
                    f"Non-tensor parameters do not match for key {key}"
                )
                new_params[key] = self.param_dict[key]
            else:
                new_params[key] = self.param_dict[key] + other.param_dict[key]

        return type(self)(param_dict=new_params, device=self.device)

    def expand(self, batch_size) -> "BatchParams":
        """
        Expands parameters with batch_size 1 to a larger batch size.

        Args:
            batch_size : int, new batch size
        """
        assert self.batch_size == 1, f"Batch size must be 1 to expand, here is {self.batch_size}"

        new_params = {}

        for key in self.param_dict.keys():
            update = self.param_dict[key]
            if not isinstance(update, torch.Tensor):
                new_params[key] = update
            else:
                n_d = len(update.shape) - 1
                new_params[key] = update.repeat(batch_size, *([1] * n_d))

        return type(self)(param_dict=new_params, device=self.device)

    def cat(self, other: "BatchParams") -> "BatchParams":
        """
        Concatenates two sets of parameters together.
        """
        assert self.k_size == other.k_size, "Kernel sizes do not match"
        assert self.device == other.device, f"Devices do not match, got {self.device} and {other.device}"
        new_params = {}
        for key in self.param_dict.keys():
            if not isinstance(self.param_dict[key], torch.Tensor):
                assert self.param_dict[key] == other.param_dict[key], (
                    f"Non-tensor parameters do not match for key {key}"
                )
                new_params[key] = self.param_dict[key]
            else:
                new_params[key] = torch.cat([self.param_dict[key], other.param_dict[key]], dim=0)

        return type(self)(param_dict=new_params, device=self.device)

    def mutate(self, magnitude=0.02, rate=0.1, frozen_keys=[]) -> "BatchParams":
        """
        Mutates the parameters by a small amount.

        Args:
            magnitude : float, magnitude of the mutation
            rate : float, will change a parameter with this rate
            frozen_keys : list of str, keys to not mutate
        """
        keys = list(self.param_dict.keys())

        new_params = {}
        for key in keys:
            if key not in frozen_keys and isinstance(self.param_dict[key], torch.Tensor):
                tentative = self.param_dict[key] * (
                    1 + magnitude * torch.randn_like(self.param_dict[key], dtype=torch.float32)
                )

                new_params[key] = torch.where(
                    torch.rand_like(tentative) < rate, tentative, self.param_dict[key]
                )
            else:
                new_params[key] = self.param_dict[key]

        return type(self)(param_dict=new_params, device=self.device)


class LeniaParams(BatchParams):
    """
    Class handling Lenia parameters

    Keys:
        'k_size' : odd int, size of kernel used for computations
        'mu' : (B,C,C) tensor, mean of growth functions
        'sigma' : (B,C,C) tensor, standard deviation of the growth functions
        'beta' :  (B,C,C, # of cores) float, max of the kernel cores
        'mu_k' : (B,C,C, # of cores) [0,1.], location of the kernel cores
        'sigma_k' : (B,C,C, # of cores) float, standard deviation of the kernel cores
        'weights' : (B,C,C) float, weights for the growth weighted sum
    """

    def __init__(
        self, param_dict=None, from_file=None, k_size=None, batch_size=None, channels=3, device="cpu"
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
                batch_size=batch_size, num_channels=channels, k_size=k_size, device=device
            ).param_dict  # dis very ugly but not sure how to do it better
            super().__init__(param_dict=param_dict, device=device)
        else:
            super().__init__(param_dict=param_dict, from_file=from_file, batch_size=batch_size, device=device)

        assert self.param_dict.keys() == {"k_size"} or set(self.param_dict.keys()) == {
            "k_size",
            "mu",
            "sigma",
            "beta",
            "mu_k",
            "sigma_k",
            "weights",
        }, f"Invalid parameter dictionary, got keys : {self.param_dict.keys()}"

        self._sanitize()
        self.to(device)

    def _sanitize(self):
        """
        Sanitizes the parameters by clamping them to valid values.
        """
        self.mu = torch.clamp(self.mu, 0, 2)
        self.sigma = torch.clamp(self.sigma, 0, None)
        # self.beta = torch.clamp(self.beta,0,1)
        self.mu_k = torch.clamp(self.mu_k, 0, 2)
        self.sigma_k = torch.clamp(self.sigma_k, 0, None)
        self.weights = torch.clamp(self.weights, 0, None)

    @staticmethod
    def default_gen(batch_size, num_channels=3, k_size=None, device="cpu"):
        """

        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        mu = 0.7 * torch.rand((batch_size, num_channels, num_channels), device=device)
        sigma = (
            mu
            / (math.sqrt(2 * math.log(2)))
            * 0.8
            * torch.rand((batch_size, num_channels, num_channels), device=device)
            + 1e-4
        )

        params = {
            "k_size": k_size if k_size is not None else 25,
            "mu": mu,
            "sigma": sigma,
            "beta": torch.rand((batch_size, num_channels, num_channels, 3), device=device),
            "mu_k": 0.5 + 0.2 * torch.randn((batch_size, num_channels, num_channels, 3), device=device),
            "sigma_k": 0.05
            * (
                1
                + torch.clamp(
                    0.3 * torch.randn((batch_size, num_channels, num_channels, 3), device=device), min=-0.9
                )
                + 1e-4
            ),
            "weights": torch.rand(batch_size, num_channels, num_channels, device=device)
            * (1 - 0.8 * torch.diag(torch.ones(num_channels, device=device))),
        }

        return LeniaParams(params, device=device)

    @staticmethod
    def random_gen(batch_size, num_channels=3, k_size=None, device="cpu"):
        """
        Full random generation
        Args:
            batch_size : number of parameters to generate
            device : device on which to generate the parameters

        Returns:
            dict of batched parameters
        """
        mu = torch.rand((batch_size, num_channels, num_channels), device=device)
        sigma = torch.rand((batch_size, num_channels, num_channels), device=device) + 1e-4

        params = {
            "k_size": k_size if k_size is not None else 25,
            "mu": mu,
            "sigma": sigma,
            "beta": torch.rand((batch_size, num_channels, num_channels, 3), device=device),
            "mu_k": torch.rand((batch_size, num_channels, num_channels, 3), device=device),
            "sigma_k": 0.05
            * (
                1
                + torch.clamp(
                    0.3 * torch.randn((batch_size, num_channels, num_channels, 3), device=device), min=-0.9
                )
                + 1e-4
            ),
            "weights": torch.rand(batch_size, num_channels, num_channels, device=device)
            * (1 - 0.8 * torch.diag(torch.ones(num_channels, device=device))),
        }

        return LeniaParams(params, device=device)
