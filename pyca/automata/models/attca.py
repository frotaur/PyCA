from ..automaton import Automaton
import torch, pygame
import torch.nn.functional as F
import colorsys


class AttCA(Automaton):
    """
    2D Cellular Automaton, with outer-holistic rules and binary values.
    The state of a pixel in the next generation depends on its own state and
    the sum of the values of its neighbors.
    """

    def __init__(self, size, random=False, device="cpu"):
        """
        Params :
        size : tuple, size of the automaton
        random : bool, if True, the initial state of the automaton is random. Otherwise, a small random square is placed in the middle.
        """
        super().__init__(size)

        self.m0_num, self.m1_num = self.get_random_rule_code()
        self.m0, self.m1 = self.get_rule_from_code(self.m0_num, self.m1_num)

        self.random = random
        self.device = device

        self.world = torch.zeros((1, self.h, self.w), dtype=torch.int, device=device)
        self.reset()

        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=torch.float)[
            None, None, :, :
        ]  # (1,1,3,3) as required by Pytorch

        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=torch.float)[
            None, None, :, :
        ]  # (1,1,3,3) as required by Pytorch

        self.side_kernels = (
            torch.tensor(
                [
                    [[1, 2, 1], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [1, 2, 1]],
                    [[1, 0, 0], [2, 0, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 0, 2], [0, 0, 1]],
                ]
            )
            .to(device)
            .unsqueeze(1)
        )  # (4,1,3,3) UP, DOWN, LEFT, RIGHT

        self.change_highlight_color()
        self.decay_speed = 0.1
        self.threshold = 2.0

        self._worldmap = self._worldmap.to(device)

        self.alternating = False  # Used to alternate between two rules
        self.count = 0  # Used to count the number of steps

    def change_highlight_color(self):
        """
        Changes the highlight color, gets random hue
        """
        hue = torch.rand(1).item()
        light = 0.5
        saturation = 0.5
        self.highlight_color = torch.tensor(
            colorsys.hls_to_rgb(hue, saturation, light), dtype=torch.float, device=self.device
        )


    def reset(self):
        """
        Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3, self.h, self.w), device=self.device)

        if self.random:
            self.world = self.get_init_mat(0.5)
        else:
            self.world = torch.zeros_like(self.world, dtype=torch.int, device=self.device)
            self.world[:, self.h // 2 - 1 : self.h // 2 + 1, self.w // 2 - 1 : self.w // 2 + 1] = torch.randint(
                0, 2, (1, 2, 2), device=self.device
            )

    def draw(self):
        """
        Updates the worldmap with the current state of the automaton.
        """
        if(not self.alternating):
            self.count=0
        
        if(self.count % 2 == 0):
            echo = torch.clamp(
                self._worldmap - self.decay_speed * self.highlight_color[:, None, None], min=0, max=1
            ).to(self.device)
            self._worldmap = torch.clamp(
                self.world.expand(3, -1, -1).to(dtype=torch.float) + echo, min=0, max=1
            ).to(self.device)
            self.count += 1
        else:
            self.count+=1

    def process_event(self, event, camera=None):
        """
        DEL -> re-initializes the automaton
        I -> toggles initialization between noise and a single dot
        N -> pick a new random rule
        Z -> change the highlight color
        A -> toggle flash-preventing mode
        UP -> longer-lasting highlights
        DOWN -> shorter-lasting highlights
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.reset()
            if event.key == pygame.K_i:
                self.random = not self.random
            if event.key == pygame.K_n:
                # Picks a random rule
                self.set_random_rule()
                print(f"rule :  s:{self.m0_num}, b:{self.m1_num}")
            if event.key == pygame.K_a:
                self.alternating = not self.alternating
            if event.key == pygame.K_z:
                self.change_highlight_color()
            if event.key == pygame.K_UP:
                self.decay_speed = max(self.decay_speed - 0.1 * self.decay_speed, 0.005)
            if event.key == pygame.K_DOWN:
                self.decay_speed = min(0.1 * self.decay_speed + self.decay_speed, 3)

    def change_num(self, s_num: int | str, b_num: int | str):
        """
        Changes the rule of the automaton to the one specified by s_num and b_num
        """
        if isinstance(s_num, str):
            s_num = self.get_num_from_rule(s_num)
        if isinstance(b_num, str):
            b_num = self.get_num_from_rule(b_num)

        self.s_num = s_num
        self.b_num = b_num
        self.reset()

    def step(self):
        # Generate tensors for all 8 neighbors
        padded_world = F.pad(
            self.world[None].to(torch.float), (1, 1, 1, 1), mode="circular"
        )  # (1,1,H+2,W+2) circular padded
        all_convs = torch.cat([self.sobel_x, self.sobel_y, self.side_kernels], dim=0)  # (6,1,3,3)
        convoluted = F.conv2d(padded_world, all_convs)  # All convolutions, sx, sy and the neighborhoods
        convoluted = convoluted.transpose(0,1) # (1,6,H,W)
        sx, sy, neighbors = convoluted[0], convoluted[1], convoluted[2:]  # (1,H,W), (1,H,W), (1,4,H,W)

        sx = torch.where(
            sx >= self.threshold, 1, torch.where(sx <= -self.threshold, -1, 0)
        )  # Apply threshold to sobel x
        sy = torch.where(
            sy >= self.threshold, 1, torch.where(sy <= -self.threshold, -1, 0)
        )  # Apply threshold to sobel y

        eff_counts = (
            neighbors[0] * (sy == 1)
            + neighbors[1] * (sy == -1)
            + neighbors[2] * (sx == 1)
            + neighbors[3] * (sx == -1)
            + (neighbors[2] + neighbors[3]) * (sx == 0)
            + (neighbors[0] + neighbors[1]) * (sy == 0)
        )  # (1,H,W)
        n_sizes = (
            4 * ((sy == 1) | (sy == -1)) + 4 * ((sx == 1) | (sx == -1)) + 8 * (sx == 0) + 8 * (sy == 0)
        )  # (1,H,W) Sizes of the effective neighborhoods

        for_lookup = torch.stack(
            [eff_counts, n_sizes], dim=0
        ).to(torch.int)  # (2,1,H,W) contains 2-uples <effective count, neighborhood size>
        for_lookup = for_lookup.to(torch.int)  # Convert to int for indexing
        # Get the next state of the world
        self.world = torch.where(self.world == 0, self.m0[for_lookup[1], for_lookup[0]], self.m1[for_lookup[1], for_lookup[0]])

    def get_init_mat(self, rand_portion):
        """
        Get initialization matrix for CA

        Params :
        rand_portion : float, portion of the screen filled with noise.
        """
        batched_size = torch.tensor([self.h, self.w])
        randsize = (batched_size * rand_portion).to(dtype=torch.int16)  # size of the random square
        randstarts = (batched_size * (1 - rand_portion) / 2).to(
            dtype=torch.int16
        )  # Where to start the index for the random square

        randsquare = torch.where(
            torch.randn(*randsize.tolist()) > 0, 1, 0
        )  # Creates random square of 0s and 1s

        init_mat = torch.zeros((self.h, self.w), dtype=torch.int16)
        init_mat[randstarts[0] : randstarts[0] + randsize[0], randstarts[1] : randstarts[1] + randsize[1]] = (
            randsquare
        )
        init_mat = init_mat[None].to(torch.int16)

        return init_mat.to(self.device)  # (1,H,W)

    def set_random_rule(self):
        """
            Sets a random rule for the automaton.
        """
        self.m0, self.m1 = self.get_random_rule()
        self.m0_num, self.m1_num = self.get_code_from_rule(self.m0, self.m1)


    @staticmethod
    def get_random_rule():
        """possible neighbourhood kernels:
        1 2 2   2 2 2   2 2 2
        0 - 2   2 - 2   2 - 2
        0 0 1   1 0 1   2 2 2

        total sizes: 8, 12, 16

        The rule is represented by two update matrices, M0 is used when middle cell is 0, M1 is used when middle cell is 1.
        Position (i, j) of the matrix represents i active cells in a noighborhood size j; M[i, j] = x means the next state will be x
        Thus, M0 and M1 have shape 17x17. The only relevant rows have indices 8, 12, 16. For the 8th row, the only relevant columns are 0, ..., 8. etc.
        """
        # possible neighborhoods are 

        M0 = torch.zeros((17, 17), dtype=torch.int)
        row_8 = torch.randint(0, 2, size=(9,), dtype=torch.int32)
        row_12 = torch.randint(0, 2, size=(13,), dtype=torch.int32)
        row_16 = torch.randint(0, 2, size=(17,), dtype=torch.int32)

        M0[8, :9] = row_8
        M0[12, :13] = row_12
        M0[16, :] = row_16



        M1 = torch.zeros((17, 17), dtype=torch.int)
        row_8 = torch.randint(0, 2, size=(9,), dtype=torch.int32)
        row_12 = torch.randint(0, 2, size=(13,), dtype=torch.int32)
        row_16 = torch.randint(0, 2, size=(17,), dtype=torch.int32)

        M1[8, :9] = row_8
        M1[12, :13] = row_12
        M1[16, :] = row_16


        return M0, M1

    @staticmethod
    def get_random_rule_code():
        """
        Get a random rule code, represented by two integers.
        Each integer is in the range [0, 549755813887] (2^39 - 1).
        """
        M0, M1 = AttCA.get_random_rule()

        return AttCA.get_code_from_rule(M0, M1)

    @staticmethod
    def dec_to_bin(n, length=8):
        """
        Convert a decimal number to binary representation with a fixed length.
        """
        return format(n, f'0{length}b')

    @staticmethod
    def bin_to_dec(b):
        """
        Convert a binary string to decimal.
        """
        return int(b, 2)

    @staticmethod
    def get_rule_from_code(m0,m1):
        """
        Convert a code to a rule represented by two matrices M0 and M1.
        The code are two integers from 0 to 549755813887 that are converted into two binary strings m0, m1, each of length 39 (=9+13+17).
        m0 = row8||row12||row16, m1 = row8||row12||row16
        """
        m0 = AttCA.dec_to_bin(m0, 39)
        m1 = AttCA.dec_to_bin(m1, 39)

        # print(m0)
        # print(m1)

        M0 = torch.zeros((17, 17), dtype=torch.int)
        M0[8, :9] = torch.tensor([int(x) for x in m0[:9]], dtype=torch.int)
        M0[12, :13] = torch.tensor([int(x) for x in m0[9:22]], dtype=torch.int)
        M0[16, :] = torch.tensor([int(x) for x in m0[22:]], dtype=torch.int)

        M1 = torch.zeros((17, 17), dtype=torch.int)
        M1[8, :9] = torch.tensor([int(x) for x in m1[:9]], dtype=torch.int)
        M1[12, :13] = torch.tensor([int(x) for x in m1[9:22]], dtype=torch.int)
        M1[16, :] = torch.tensor([int(x) for x in m1[22:]], dtype=torch.int)

        return M0, M1

    @staticmethod
    def get_code_from_rule(M0, M1):
        """
        Convert two matrices M0 and M1 to a code represented by two integers.
        The matrices are expected to be of shape (17, 17).
        """
        m0 = ''.join([str(x) for x in M0[8, :9].tolist()]) + \
            ''.join([str(x) for x in M0[12, :13].tolist()]) + \
            ''.join([str(x) for x in M0[16, :].tolist()])

        m1 = ''.join([str(x) for x in M1[8, :9].tolist()]) + \
            ''.join([str(x) for x in M1[12, :13].tolist()]) + \
            ''.join([str(x) for x in M1[16, :].tolist()])

        return AttCA.bin_to_dec(m0), AttCA.bin_to_dec(m1)