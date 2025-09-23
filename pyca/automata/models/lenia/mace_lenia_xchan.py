import torch
import pygame

from .macelenia import MaCELenia
from .lenia import Lenia
from pyca.interface import LabeledSlider

class MaCELeniaXChan(MaCELenia):
    """
        MaCELenia with cross channel step
    """

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
        """
        Args:
            size : tuple, (B,H,W) size of the automaton
            dt : float, time step size
            num_channels : int, number of channels
            params : dict, parameters of the automaton
            state_init : tensor, initial state of the automaton
            device : str, device to use
        """

        super().__init__(
            size,
            num_channels,
            params,
            state_init,
            has_food=has_food,
            sense_food=sense_food,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        self._beta = 6  # default temperature is 6 for this one
        self.alpha = 0.03
        self.params["alpha"] = self.alpha

        self.alpha_slider = LabeledSlider(0.0, 0.3,title="alpha", fract_size=(0.07, 0.18), precision=2, initial_value=self.alpha)
        self.register_component(self.alpha_slider, custom_size=True)

    def update_params(self, params, k_size_override=None):
        super().update_params(params, k_size_override=k_size_override)
        if "alpha" in params:
            self.alpha = params["alpha"]

    def step(self):
        """
        Steps the alife model by one time step
        """
        Aff = self._mace_step(sense_food=self.sense_food)
        if self.has_food:
            self._food_step()
        self._cross_chan_step(Aff)  # (B,C,H,W) cross channel step

    def _cross_chan_step(self, Aff):
        """Performs the cross channel step, given the affinity matrix"""
        max_Aff = torch.max(Aff, dim=1, keepdim=True)[0]
        Aff_shifted = self.b * (Aff - max_Aff)
        numerator = torch.exp(Aff_shifted)
        Aff_c = numerator / (numerator.sum(dim=1, keepdim=True))

        target_cross_c_masses = self.state.sum(dim=1, keepdim=True) * Aff_c
        self.state = self.state - (self.state - target_cross_c_masses) * self.alpha

    def process_event(self, event, camera=None):
        """
        LEFT/RIGHT: change alpha
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.alpha -= 0.02
                self.params["alpha"] = self.alpha
            if event.key == pygame.K_RIGHT:
                self.alpha += 0.02
                self.params["alpha"] = self.alpha

        for compo in self.changed_components:
            if compo == self.alpha_slider:
                self.alpha = self.alpha_slider.value
                self.params["alpha"] = self.alpha
    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        return super().get_string_state() + f"alpha: {self.alpha:.2f}"
