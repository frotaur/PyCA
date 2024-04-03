from Automaton import Automaton
import torch, pygame

class CA2D(Automaton):
    """
        2D outer holistic cellular automaton, with two states.
    """

    def __init__(self, size, s_num='23', b_num='3', random=False):
        """
            Params :
            size : tuple, size of the automaton
            s_num : str, rule for survival
            b_num : str, rule for birth
            random : bool, if True, the initial state of the automaton is random. Otherwise, a small random square is placed in the middle.
        """
        super().__init__(size)

        self.s_num = self.get_num_rule(s_num) # Translate string to number form
        self.b_num = self.get_num_rule(b_num) # Translate string to number form
        self.random = random

        self.world = torch.zeros((self.h,self.w),dtype=torch.int)
        self.reset()


    def get_num_rule(self,num):
        """
            Get the rule number for the automaton
        """
        rule_out = 0
        if(num!=''): # If the num is empty, the associated number is 0
            rule_out = sum([2**int(d) for d in num])

        return rule_out

    def reset(self):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))

        if(self.random):
            self.world = self.get_init_mat(0.5)
        else:
            self.world = torch.zeros_like(self.world,dtype=torch.int)
            self.world[self.w//2-1:self.w//2+1,self.h//2-1:self.h//2+1]=torch.ranint(0,2,(2,2))
    
    def draw(self):
        """
            Updates the worldmap with the current state of the automaton.
        """
        self._worldmap = self.world[None,:,:].expand(3,-1,-1).to(dtype=torch.float)
        
    def process_event(self, event, camera=None):
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                self.reset() 
            if(event.key == pygame.K_n):
                # Picks a random rule
                b_rule = torch.randint(0,2**8,(1,)).item()
                s_rule = torch.randint(0,2**8,(1,)).item()
                self.change_num(s_rule,b_rule)
                print('rule : ', (s_rule,b_rule))

    def change_num(self,s_num : int, b_num : int):
        """
            Changes the rule of the automaton to the one specified by s_num and b_num
        """
        self.s_num = s_num
        self.b_num = b_num
        self.reset()
    
    def step(self):
        # Generate tensors for all 8 neighbors
        w, e = self.world.roll(-1, 0), self.world.roll(1, 0) 
        n, s = self.world.roll(-1, 1), self.world.roll(1, 1) 
        sw, se = w.roll(1, 1), e.roll(1, 1)
        nw, ne = w.roll(-1, 1), e.roll(-1, 1)

        count = w + e + n + s + sw + se + nw + ne

        self.world = torch.where(self.world==1,self.get_nth_bit(self.s_num,count),self.get_nth_bit(self.b_num,count)).to(torch.int)

    def get_nth_bit(self,num, s):
        """
            Get the nth bit of the number num
        """
        return (num >> s) & 1

    def get_init_mat(self,rand_portion):
        """
            Get initialization matrix for CA

            Params : 
            rand_portion : float, portion of the screen filled with noise.
        """
        batched_size = torch.tensor([self.h,self.w])
        randsize = (batched_size*rand_portion).to(dtype=torch.int16) # size of the random square
        randstarts = (batched_size*(1-rand_portion)/2).to(dtype=torch.int16) # Where to start the index for the random square

        randsquare = torch.where(torch.randn(*randsize.tolist())>0,1,0) # Creates random square of 0s and 1s

        init_mat = torch.zeros((self.h,self.w),dtype=torch.int16)
        init_mat[randstarts[0]:randstarts[0]+randsize[0],
        randstarts[1]:randstarts[1]+randsize[1]] = randsquare
        init_mat = init_mat.to(torch.int16)


        return init_mat # (B,H,W)