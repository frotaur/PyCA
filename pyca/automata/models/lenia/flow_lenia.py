import torch, torch.nn
import pygame
from .lenia import Lenia
import random

def sobel_w(x,device="cpu"):
    """
        Args:
            x: (B,C,H,W) tensor, where B is the batch size, C is the number of channels, H is the height and W is the width.
    """
    # In a (H,W) tensor, the derivative with width is differentiating with columns
    k_w = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).tile(
        (x.shape[1], 1, 1, 1)
    )

    sw = torch.nn.functional.conv2d(x, k_w, groups=x.shape[1], stride=1, padding="same")

    return sw


def sobel_h(x,device="cpu"):
    """
        Args:
            x: (B,C,H,W) tensor
    """
    k_h = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).tile(
        (x.shape[1], 1, 1, 1)
    ) # (C, 1, k_size, k_size)
    sh = torch.nn.functional.conv2d(x, k_h, groups=x.shape[1], stride=1, padding="same")

    return sh


def sobel(x,device="cpu"):
    """
        Args:
        x: (B,C,H,W) tensor

        Returns:
        sxy: (B,C,2,H,W) tensor, where the first channel is the sobel_y and the second channel is the sobel_x
    """
    sh = sobel_h(x.to(torch.float32),device=device)

    sw = sobel_w(x.to(torch.float32),device=device)
    sxy = torch.stack((sw, sh), dim=2) # (B,C,2,H,W) vector is derivative in (width, height), preparing for the RT which expexts those conventions
    return sxy


def construct_mesh_grid(X, Y):
    x, y = torch.arange(X), torch.arange(Y)
    mx, my = torch.meshgrid(x, y) #  mx, my are (W,H)
    pos = torch.dstack((mx, my)) + 0.5 # (W,H,2) pos[x,y] = (x,y)
    # pos = pos.permute((2,1,0))
    return pos


def construct_ds(dd, device="cpu"):
    dxs = []
    dys = []
    for dx in range(-dd, dd + 1):
        for dy in range(-dd, dd + 1):
            dxs.append(dx)
            dys.append(dy)
    dxs = torch.tensor(dxs, device=device)
    dys = torch.tensor(dys, device=device)
    return dxs, dys



class TrueRT:
    def __init__(self, W, H, dt, dd=2, sigma=0.65,device='cpu'):
        self.X = W
        self.Y = H
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.device = device
        self.pos = self.construct_mesh_grid(W, H) # (W, H, 2) tensor with positions
        self.dxs, self.dys = self.construct_ds(dd)

    def construct_mesh_grid(self, W, H):
        x, y = torch.arange(W), torch.arange(H)
        mx, my = torch.meshgrid(x, y)
        pos = torch.dstack((mx, my)) + 0.5
        return pos.to(self.device)

    def construct_ds(self, dd):
        dxs, dys = [], []
        for dx in range(-dd, dd + 1):
            for dy in range(-dd, dd + 1):
                dxs.append(dx)
                dys.append(dy)

        return dxs, dys
    

    def step(self, state, mu, dxs, dys):
        """
            Applies the RT method for neighbors defined by dxs, dys.

            Args:
                state: (B, C, W, H) tensor representing the current state.
                mu: (B, C, W, H, 2) tensor with positions to update.
                dxs: (N,) tensor of x offsets
                dys: (N,) tensor of y offsets
        """
        # Do the steps for each dx, dy pair, and sum the contributions
        # from all neighbors to the state.
        
        return torch.stack([self.step_one_dx(state, mu, dx, dy) for dx, dy in zip(dxs, dys)], dim=0).sum(dim=0)
    
    def step_one_dx(self, state, mu, dx, dy):
        """
            Applies one step of the RT method for a single dx, dy pair.
            Args:
                state: (B, C, W, H) tensor representing the current state.
                mu: (B, C, W, H, 2) tensor with positions to update.
                dx: x offset
                dy: y offset
            
            Returns: Contribution to the state coming from the dx, dy shifted neighbors
        """
        state_rolled = torch.roll(state, shifts=(dx,dy), dims=(2,3))
        mu_rolled = torch.roll(mu, shifts=(dx,dy), dims=(2,3))

        # Compute signed distance from the center of the sigma square
        # to the center of the currently considered square
        distri_center_distance = torch.abs(mu_rolled - self.pos[None,None])  # (B, C, W, H, 2)
        # Calculate the 'unclipped' and unnormalized overlap amount for both x and y directions
        overlap_amount = .5+(self.sigma - distri_center_distance) # (B, C, W, H, 2)
        overlap_area = torch.clip(overlap_amount, 0, min(2*self.sigma,1)).prod(dim=-1)
        overlap_area = overlap_area/(4*self.sigma**2)  # Normalize the overlap area to [0,1]

        return state_rolled*overlap_area # The overlap area is stolen propotionally from the state
    
    def apply_flow(self, state, flow):
        """
            Returns new state after applying flow to the current state.

            Args:
                state: (B, C,H,W) tensor representing the current state.
                flow: (B, C, 2, H, W) vectors of flow for each channel
        """
        flow_perm = torch.einsum("bcfhw->bcwhf", flow)  # (B,C,2,H,W) to (B,C,W,H,2)
        state_perm = torch.einsum("bchw->bcwh", state)  # (B,C,H,W) to (B,C,W,H)
        max_flow = self.dd - self.sigma

        mu = self.pos[None, None] + torch.clip(self.dt * flow_perm, -max_flow, max_flow)  # (B,C,W,H,2)

        new_state = self.step(state_perm, mu, self.dxs, self.dys)  # (B,C,W,H) tensor with new state
        new_state = torch.einsum("bcwh->bchw", new_state)

        return new_state
       
class FlowLenia(Lenia):
    """Pytorch port of mass conserving FlowLenia"""

    def __init__(
        self,
        size,
        dt=0.1,
        num_channels=3,
        params=None,
        state_init=None,
        device="cpu",
        dd=2,
        sigma_rt=0.65,
        has_food=False,
        interest_files=None,
        save_dir="."
    ):
        if(len(size) == 2):
            size = (1,)+size
        self.dd = dd
        self.sigma_rt = sigma_rt
        self.theta_x = 2
        self.n = 2
        self.rt = TrueRT(size[2], size[1], dt, dd=self.dd,  sigma=self.sigma_rt, device=device)

        # self.rt = ReintegrationTracker(size[1], size[2], dt, dd=self.dd, sigma=self.sigma_rt)
        self.has_food = has_food
        super().__init__(
            size,
            dt,
            num_channels,
            params,
            state_init,
            device=device,
            interest_files=interest_files,
            save_dir=save_dir
        )

        self.Aff = self.compute_affinity()
        self.params.dd = self.dd
        self.params.sigma_rt = self.sigma
        self.params.theta_x = self.theta_x

    def change_dd(self, dd):
        """
        Change the dd parameter of the ReintegrationTracker.
        Args:
            dd (int): new dd value
        """
        self.dd = dd
        self.rt.change_dd(dd)

    def compute_affinity(self):
        Aff = self.kernel_fftconv(self.state)  # (B,C,C,H,W) first step affinity, usual convolutions
        weights = self.weights[..., None, None]  # (B,C,C,1,1)
        Aff = (self.growth(Aff) * weights).sum(dim=1)  # (B,C,H,W) pre-exponential affinity

        grad_u = sobel(Aff,self.device)  # (B,C,2,H,W)

        grad_x = sobel(self.state.sum(dim=1, keepdims=True),self.device) # (B,1,2,H,W) gradient of the mass channel

        # added a sum over the channel in the alpha computation, as in the paper
        alpha = (
            (self.state.sum(dim=1, keepdims=True) / self.theta_x)
            ** self.n
        ).clip(0, 1)[:,:,None] # (B,C,1,H,W), broadcastable against the gradients

        F = grad_u * (1 - alpha) - grad_x * alpha # (B,C,2,H,W) final flow field
        # F= grad_u

        return F # (B,C,2,H,W)

    def update_food(self):
        """uncomment the death sections for death mechanics, but its finicky and i dont like it"""
        where_food = self.food_channel > 0  # Where the food channels are
        where_contact = (
            self.state.sum(dim=1)[:, None, :, :] > 0.1
        )  # Where the eating channel is, we could amke this dynamic, 0.1 is the threshold for eating
        death = (
            (self.state.sum(dim=1)[:, None, :, :] < 0.01) & (self.state.sum(dim=1)[:, None, :, :] > 0)
        ) * self.state  # death of the feeding channel, very finicky

        overlap = where_food & where_contact  # where the channels overlap
        transfer = (
            torch.ones_like(where_food) * overlap * 0.03
        )  # How much to increase / deacrease the mass currently set to 0.01
        self.state += transfer  # Lenia mass increase
        self.state -= death
        self.food_channel -= transfer  # Food mass deacrease
        self.food_channel += death.sum(dim=1)[:, None, :, :] / 3

    @torch.no_grad()
    def step(self):
        Aff = self.compute_affinity()  # (B,C,2,H,W) flow field

        self.state = self.rt.apply_flow(self.state, Aff)
        if self.has_food:
            self.update_food()
        self.frames += 1  # Increment frame count


    def process_event(self, event, camera=None):
        """
        UP -> Increase temperature
        DOWN -> Decrease temperature
        RIGHT-> Increase sigma
        LEFT -> Decrease sigma
        """
        super().process_event(event, camera)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.theta_x += 0.2
            if event.key == pygame.K_DOWN:
                self.theta_x -= 0.2

            if event.key == pygame.K_RIGHT:
                self.sigma_rt += 0.02
                self.rt.sigma = self.sigma_rt
            if event.key == pygame.K_LEFT:
                self.sigma_rt -= 0.02
                self.rt.sigma = self.sigma_rt

            if event.key ==pygame.K_t:
                if(pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.dt-= 0.01
                    print('Decreasing dt to', self.dt)
                else:
                    self.dt += 0.01
                    print('Increasing dt to', self.dt)
            if event.key == pygame.K_d:
                if(pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.dd -= 1
                    self.change_dd(self.dd)
                    print('Decreasing dd to', self.dd)
                else:
                    self.dd += 1
                    self.change_dd(self.dd)
                    print('Increasing dd to', self.dd)
    process_event.__doc__ = Lenia.process_event.__doc__.rstrip("\n") + process_event.__doc__.lstrip(
        "\n"
    )  # Hack to append the docstring of MCLenia.process_event

    def get_string_state(self):
        return (
            super().get_string_state() + f" mass: {self.state.sum().item():.2f}, theta_x: {self.theta_x:.2f}, sigma: {self.sigma_rt:.2f}"
        )

    def random_food_chan(self, num_spots=100, food_size=5):
        """
        Returns a food channel with num_spots of food of size food_size
        Args :
            num_spots : int, number of food spots
            food_size : int, size of the food spots

        Returns :
            food_chan : tensor, (B,1,H,W) food channel
        """
        places = [
            [random.randint(food_size, self.h - food_size), random.randint(food_size, self.w - food_size)]
            for _ in range(num_spots)
        ]
        food_chan = torch.zeros((self.batch, 1, self.h, self.w), device=self.device)
        for place in places:
            food_chan[
                :, :, place[0] - food_size : place[0] + food_size, place[1] - food_size : place[1] + food_size
            ] = 1

        return food_chan

    def set_init_fractal(self):
        super().set_init_fractal()
        if self.has_food:
            self.food_channel = self.random_food_chan()  # (B,1, H,W)

    def set_init_perlin(self, wavelength=None):
        super().set_init_perlin(wavelength)
        if self.has_food:
            self.food_channel = self.random_food_chan()  # (B,1, H,W)

    def set_init_circle(self, fractal=False, radius=None):
        super().set_init_circle(fractal, radius)
        if self.has_food:
            self.food_channel = self.random_food_chan()  # (B,1, H,W)

    @torch.no_grad()
    def draw(self):
        """
        Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"

        toshow = self.state[0].clone()  # (C,H,W), pygame conversion done later

        if self.C == 1:
            toshow = toshow.repeat(3, 1, 1)  # (3,H,W)
        elif self.C == 2:
            toshow = torch.cat([toshow, torch.zeros_like(toshow)], dim=0)  # (3,H,W)
        else:
            toshow = toshow[:3, :, :]  # (3,H,W)

        if self.has_food:
            toshow[:, :, :] += self.food_channel[0]  # (1,H,W)

        if self.display_kernel == True:
            kern = self.compute_ker()  # (C,3,k_size,k_size)
            for i in range(kern.shape[0]):
                toshow[:, self.h - self.k_size : self.h, i * self.k_size : (i + 1) * self.k_size] = kern[
                    i
                ].cpu()

        self._worldmap = torch.clamp(toshow, 0.0, 1.0)
