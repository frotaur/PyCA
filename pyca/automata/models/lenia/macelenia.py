import torch, torch.nn.functional as F
import pygame
import showtens

from pyca.interface.ui_components.Slider import LabeledSlider
from .lenia import Lenia
import random
import math
from pyca.interface import Toggle, Slider

class MaCELenia(Lenia):
    """
    Mass conserving Lenia-like Alife model
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
            size : tuple, (B,H,W) or (H,W) of ints, size of the automaton
            num_channels : int, number of channels
            params : dict, parameters of the automaton
            state_init : tensor, initial state of the automaton
            device : str, device to use
        """
        if(len(size) == 2):
            size = (1,)+size
        self.has_food = has_food  # Needed for initialization
        self.initial_food = 0.1* size[1] * size[2]  # Amount of food to add at initialization
        self.sense_food = sense_food

        super().__init__(
            size,
            dt=0.1,  # dt does not matter this MaCE update
            num_channels=num_channels,
            params=params,
            state_init=state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir,
        )

        self._beta = 8
        self.Aff = self._compute_affinity()
        self.show_batch = 0
        self.cum_loss_mass = torch.zeros(self.batch, device=device)
        self.show_all = False
        self.show_all_override = False

        kernel_size = 7
        self.smear_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device) / (
            kernel_size * kernel_size
        )  # (1,1,kernel_size,kernel_size)

        self.add_speed = 1.
        
        # GUI components
        self.tog_food_off = Toggle("Food: OFF", "Food: ON")
        self.register_component(self.tog_food_off)
        self.beta_slider = LabeledSlider(min_value=0., max_value=15., fract_size=(0.07, 0.18), title='Beta/Temp', initial_value=self.b)
        self.register_component(self.beta_slider, custom_size=True)

    def step(self, sense_food=None):
        """
        Steps the alife model by one time step

        Args: sense_food: overrides self.sense_food, if True, the model will sense food
        """
        if sense_food is not None:
            self._mace_step(sense_food=sense_food)
        else:
            # Perform the MaCE step, computing affinity and redistributing mass
            self._mace_step(sense_food=self.sense_food)

        if self.has_food:
            # If food is active, we perform the decay and mass redistribution step
            self._food_step()
        self.frames += 1  # Increment frame count

    def _mace_step(self, sense_food=False):
        """
        Performs the Mace Step of the model,
        and returns the affinity tensor for potential further use.

        Args:
            sense_food : bool, if True, the model will sense food
            and update the affinity tensor accordingly
        Returns:
            Aff : (B,C,H,W), affinity tensor of the model
        """
        B, C, H, W = self.state.shape

        # Compute affinity with growth function of Lenia
        Aff = self._compute_affinity(sense_food=sense_food)  # (B,C,H,W) affinity matrix
        expAff = torch.exp(self.b * Aff)  # Exponentiate with beta

        # Unfold expAff to prepare the computation of normalization Z
        Z = F.pad(expAff, (1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        Z = F.unfold(Z, kernel_size=(3, 3)).reshape(B, C, 9, H, W)  # (B,C*9,H,W)
        Z = Z.sum(dim=2)  # (B,C,H,W) local affinity normalization tensor

        to_give = self.state / Z  # normalized mass, ready to be portioned according to the expAff
        to_give = F.pad(to_give, (1, 1, 1, 1), mode="circular")  # (B,C,H+2,W+2) for the (3,3) kernel
        to_give = F.unfold(to_give, kernel_size=(3, 3)).reshape(
            B, C, 9, H, W
        )  # (B,C,9,H,W), unfold again to distribute mass to all 9 neighbors
        self.state = (expAff[:, :, None] * to_give).sum(
            dim=2
        )  # (B,C,H,W) result of the redistribution of mass

        return Aff  # (B,C,H,W), return affinity if needed later (e.g. for cross channel step)

    def _compute_affinity(self, sense_food=False):
        """
        Computes the affinity tensor of the model, using the Lenia way of computing the growth value.
        With sense_food, the food will alter the affinity value to increase it if it's present in one of the kernels.

        Args:
            sense_food : bool, if True, the model will sense food
            and update the affinity tensor accordingly
        Returns:
            Aff : (B,C,H,W) tensor, affinity
        """
        ### Alternative way of sensing food : adding food channel to the red channel of the state
        ### This makes it so sometimes, matter is repelled by food, so its a selection process on the parameters
        ## --------------------------------------------------------------------------- ##
        # if sense_food and self.has_food:
        # a = self.state.clone()
        # a[:,0:1,...] += self.food_channel # Add food channel to the state 'r' channel, for sensing
        # Aff = self.kernel_fftconv(a)  # (B,C,C,H,W) first step affinity, usual convolutions
        # else:
        #     Aff = self.kernel_fftconv(self.state)
        ## --------------------------------------------------------------------------- ##

        Aff = self.kernel_fftconv(
            self.state
        )  # Compute the convolution of Lenia kernel with the state, (B,C,C,H,W)
        weights = self.weights[..., None, None]  # (B,C,C,1,1)
        Aff = (self.growth(Aff) * weights).sum(
            dim=1
        )  # (B,C,H,W) Proper affinity, by applying growth function and summing contributions

        if self.has_food and sense_food:  # Comment this if using alternative sensing (above)
            food_aff = (self.food_channel > 0).float().expand(-1, self.C, -1, -1)  # (B,1,H,W) food channel
            # Optional : increase affinity according also to how much food is sensed
            food_aff = food_aff + self.kernel_fftconv(food_aff).sum(dim=1)  # (B,1,H,W) food channel
            # Hardcoded for now, but remove affinity when matter is too low, so it cant eat
            Aff = Aff + (food_aff)  # (B,C,H,W) food affinity

        return Aff  # (B,C,H,W) Affinity tensor

    def _food_step(self):
        """
        Performs the mass decay, food eating and food generation step. Updates self.state.
        Lots of things were tried here, unclear what's the best way to decay/consume/redistribute food
        to promote intrisinc competition in an equilibrated way! Experiment at will.
        """
        if self.has_food:  # Only do something if food is active
            self._decay_and_distribute()  # Decay the state and redistribute the mass to the food

            self._consume_food(min_density=0.3 if self.sense_food else 0.1, transfer_rate=0.1, death_enabled=False)

    def _decay_and_distribute(self):
        """
        Decays the state and redistributes the mass to the food channel.
        """
        food_amount = 200  # Amount of food redistributed per step

        # --- Decay 1 --- Exponential decay + small constant decay. Quite harsh, work best with full window initialization
        # allowable_decay = torch.minimum(self.state, self.state*0.003 + torch.full_like(self.state, 0.0003))
        # --- Decay 2 --- Same as before, but with a minimal value. Allows for a thin 'veil' of mass to spread, which helps solitons move around
        # allowable_decay = torch.where(self.state>0.01,self.state*0.0007+torch.full_like(self.state,0.02*0.0005), torch.zeros_like(self.state))
        # --- Decay 3 --- Constant decay. Encourages more high concentrations, as proportionally they decay slower
        allowable_decay = torch.where(self.state > 0.02, 0.0003, torch.zeros_like(self.state))

        self.state = self.state - allowable_decay  # Update the state by subtracting the allowable decay

        # Compute all mass lost, when above some threshold, reintroduce the mass as food
        # Must be careful not to couple different batches, as its different worlds
        self.cum_loss_mass += (allowable_decay).view(self.batch, -1).sum(dim=1)
        update_idxs = torch.argwhere(self.cum_loss_mass >= food_amount).tolist()
        update_idxs = [p[0] for p in update_idxs]

        if update_idxs:  # Each batchs that has lost more than food_amount gets a redistribution
            self.cum_loss_mass[update_idxs] = self.cum_loss_mass[update_idxs] - food_amount
            self.food_channel = self.random_food_chan(
                food_amount=food_amount, num_spots=5, food_size=7, add_to_exisitng=True, batches=update_idxs
            )

    def _consume_food(self, min_density=0.1, transfer_rate=0.03, death_enabled=False):
        """
        Performs the food consumption step, where food is eaten and transferred to the state.
        <EXPERIMENTAL> : With death_enabled = true, it also performs a state decay, which can
        turn into food. This can be turned on manually in the code, and if so, the _decay_and_distribute
        step should be removed. Very finicky, hard to stabilize, but can lead to interesting behaviors with
        lots of effort.
        """
        where_food = self.food_channel > 0  # Where the food channels are
        where_contact = (
            self.state.sum(dim=1, keepdim=True) >= min_density
        )  # Where the eating channel is, we could amke this dynamic, 0.1 is the threshold for eating

        if death_enabled:
            death = (
                (self.state < 0.04) & (self.state > 0)
            ) * self.state  # death of the feeding channel, very finicky

        overlap = where_food & where_contact  # where the channels overlap
        transfer = torch.minimum(self.food_channel, torch.ones_like(where_food) * overlap * transfer_rate)

        self.state += transfer / 3  # Lenia mass increase

        self.food_channel -= transfer

        if death_enabled:
            self.state -= death
            self.food_channel += death.sum(dim=1, keepdim=True)

    def update_show_batch(self, dirr):
        self.show_batch = (self.show_batch + dirr) % self.batch

    @property
    def b(self):
        return self._beta

    @b.setter
    def b(self, value):
        self._beta = value

    def process_event(self, event, camera=None):
        """
        UP -> Increase temperature
        DOWN -> Decrease temperature
        PLUS -> Show next batch
        MINUS -> Show previous batch
        B -> Toggle show all batches (only if batch > 1)
        F (+shift) -> Toggle food and decay (+shift toggle food sensing)
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.b += 0.2
            if event.key == pygame.K_DOWN:
                self.b -= 0.2
            if event.key == pygame.K_KP_PLUS or event.key == pygame.K_PLUS:
                self.update_show_batch(1)
            if event.key == pygame.K_KP_MINUS or event.key == pygame.K_MINUS:
                self.update_show_batch(-1)
            if event.key == pygame.K_b:
                if(self.batch > 1):
                    self.show_all = not self.show_all
            if event.key == pygame.K_f:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.sense_food = not self.sense_food
                else:
                    self.set_food(not self.has_food)

        for component in self.changed_components:
            if component == self.tog_food_off:
                self.set_food(not self.tog_food_off.state)
            if component == self.beta_slider:
                self.b = self.beta_slider.value
    
    def set_food(self, has_food: bool):
        """
        Sets the food state of the automaton.

        Args:
            has_food : bool, if True, the automaton will have food, otherwise not
        """
        self.has_food = has_food
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food)
        else:
            self.food_channel = torch.zeros_like(self.food_channel)
        self.tog_food_off.state = not self.has_food

    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        """
        Returns the info string displayed at the bottom of the window
        """
        return (
            super().get_string_state()
            + f"tot mass: {self.total_mass():.2f}, live: {self.state.sum():.2f} beta : {self.b:.2f}, Showing Batch: {self.show_batch}"
        )

    def random_food_chan(self, food_amount, num_spots=300, food_size=5, add_to_exisitng=False, batches=[]):
        """
        Returns a food channel with num_spots of food of size food_size. Can be used to initialize the food,
        or to add some food to the existing food channel. Given the food amound, the number of spots and the spot size,
        a food density will be computed, and the food will be added to the food channel.

        Args :
            food_amount : int, amount of food to add
            num_spots : int, number of food spots
            food_size : int, size of the food spots
            add_to_exisitng : bool, if True, the food will be added to the existing food channel, otherwise, resampled from scratch
            batches : list, indices of the batches to add food to

        Returns :
            food_chan : tensor, (B,1,H,W) food channel
        """
        food_density = food_amount / (
            num_spots * food_size * food_size
        )  # food_size*food_size is the area of the food spot
        places = [
            [random.randint(food_size, self.h - food_size), random.randint(food_size, self.w - food_size)]
            for _ in range(num_spots)
        ]

        if add_to_exisitng:
            food_chan = self.food_channel.clone()
        else:
            food_chan = torch.zeros((self.batch, 1, self.h, self.w), device=self.device)
        for place in places:
            if add_to_exisitng:
                food_chan[
                    batches,
                    :,
                    place[0] - food_size // 2 : place[0] + food_size // 2 + 1,
                    place[1] - food_size // 2 : place[1] + food_size // 2 + 1,
                ] = food_density
            else:
                food_chan[
                    :,
                    :,
                    place[0] - food_size // 2 : place[0] + food_size // 2 + 1,
                    place[1] - food_size // 2 : place[1] + food_size // 2 + 1,
                ] = food_density

        return food_chan

    def set_init_fractal(self):
        super().set_init_fractal()
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food)  # (B,1, H,W)
            self.cum_loss_mass = torch.zeros(self.batch, device=self.device)

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food)  # (B,1, H,W)
            self.cum_loss_mass = torch.zeros(self.batch, device=self.device)

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if self.has_food:
            self.food_channel = self.random_food_chan(food_amount=self.initial_food)  # (B,1, H,W)
            self.cum_loss_mass = torch.zeros(self.batch, device=self.device)

    @torch.no_grad()
    def draw(self):
        """
        Draws the RGB worldmap from state.
        """
        # assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        if (not self.show_all) and (not self.show_all_override):
            toshow = self.state[self.show_batch].clone()  # (C,H,W), pygame conversion done later

            if self.C == 1:
                toshow = toshow.repeat(3, 1, 1)  # (3,H,W)
            elif self.C == 2:
                toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
            else:
                toshow = toshow[:3, :, :]  # (3,H,W)

            if self.has_food:
                toshow[:, :, :] += self.food_channel[self.show_batch]  # (1,H,W)

            if self.display_kernel:
                toshow = self._draw_kernel(toshow)

            self._worldmap = torch.clamp(toshow, 0.0, 1.0)

        else:
            mod_state = self.state.clone()
            mod_state[:, :, :, 0:5] = 1
            mod_state[:, :, :, -5:] = 1
            mod_state[:, :, 0:5, :] = 1
            mod_state[:, :, -5:, :] = 1

            if self.display_kernel == True:
                for j in range(self.batch):
                    kern = self.compute_ker(batch=j)  # (C,3,k_size,k_size)
                    for i in range(kern.shape[0]):  # TODO : bugs out if num_channels =1
                        mod_state[
                            j, :, self.h - self.k_size : self.h, i * self.k_size : (i + 1) * self.k_size
                        ] = kern[i].cpu()

            toshow = showtens.gridify(
                mod_state, max_width=self.size[1] * 2, columns=int(math.sqrt(self.batch))
            )
            if self.C == 1:
                toshow = toshow.repeat(3, 1, 1)  # (3,H,W)q
            elif self.C == 2:
                toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
            else:
                toshow = toshow[:3, :, :]  # (3,H,W)

            if self.has_food:
                toshow[:, :, :] += showtens.gridify(
                    self.food_channel, max_width=self.size[1] * 2, columns=int(math.sqrt(self.batch))
                )  # (1,H,W)

            self._worldmap = torch.clamp(toshow, 0.0, 1.0)

    def total_mass(self):
        """
        Returns the total mass of the model
        """
        if self.has_food:
            return (self.state.sum(dim=(1, 2, 3)) + self.food_channel.sum(dim=(1, 2, 3)))[0]
        else:
            return self.state.sum(dim=(1, 2, 3))[0]
