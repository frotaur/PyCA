from ..automaton import Automaton
import torch, pygame
import torch.nn.functional as F


class ConservCA(Automaton):
    """
        2D CA with conservative evolution rules
    """

    def __init__(self, size, affinity=None, device='cpu'):
        """
            Args :
            size : tuple, size of the automaton
            affinity : function, function that computes the affinity of a cell to its neighborhood
            device : str, device to use for computation
        """
        super().__init__(size)
        self.device = device

        self.c = 1

        self.kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], device=device, dtype=torch.float)[
            None, None, :, :
        ].expand(self.c,-1,-1,-1)  # (c,1,3,3) as required by Pytorch


        if(affinity is None) :
            self.affinity = self._default_affinity
        
        self.world = torch.zeros((self.c,self.h,self.w), dtype=torch.int, device=device)
        
        self.hgrid, self.wgrid = torch.meshgrid(torch.arange(self.h,device=device),torch.arange(self.w,device=device)) # (h,w)

        self.faucet=False
        self.tapestry=False

    def _count(self) :
        """
            Computes neighborhood count
        """
        pad_world = F.pad(self.world[None].float(), (1,1,1,1), mode='circular') # (1,c,h+2,w+2)

        return F.conv2d(pad_world, self.kernel,groups=self.c)[0].to(torch.int) # (c,h,w)

    def _default_affinity(self, count, world) :
        """
            Default affinity function. Count is outer-holistic. World is provided for outer-holistic computations
        """
        full_sum = count+world
        score = torch.exp(-torch.abs(full_sum.float() - 10.)/5.)+torch.exp(-torch.abs(full_sum.float() - 30.)/3.) # Arbitrary

        return score
        # return -torch.sigmoid(count.float()-5.)

    def draw(self):
        """
            Draws the automaton
        """
        clip_value = 15
        self._worldmap = torch.clamp(self.world, 0, clip_value).float()/clip_value # (c,h,w)

        if(self.c==1):
            self._worldmap=self._worldmap.expand(3,-1,-1)
    
        return self._worldmap

    def step(self) :
        """
            Performs a step of the automaton
        """
        count = self._count()
        aff = self.affinity(count, self.world) # (c,h,w)

        distri_mask = F.pad(aff, (1,1,1,1), mode='circular') # (c,h+2,w+2)
        distri_mask = F.unfold(distri_mask[None], kernel_size=(3,3), padding=0).reshape(1,self.c,9,self.h,self.w)

        maxes = torch.max(distri_mask, dim=2,keepdim=False).values # (1,c,h,w), max of each neighborhood

        value_normalization = (distri_mask == maxes[:,:,None]).int().sum(dim=2,keepdim=False) # (1,c,h,w), number of maxes in each neighborhood

        maxes = F.pad(maxes, (1,1,1,1), mode='circular') # (1,c,h+2,w+2)
        maxes = F.unfold(maxes, kernel_size=(3,3), padding=0).reshape(1,self.c,9,self.h,self.w) # (1,c,9,h,w)

        winner_mask = (aff[None,:,None] == maxes).int() # (1,c,9,h,w) For a cell, the 3*3 tensor contains 1 if it was the max of the neighborhood

        world_share = self.world[None]//value_normalization # (1,c,h,w)
        world_remain = self.world[None]%value_normalization # (1,c,h,w)

        world_share = F.pad(world_share, (1,1,1,1), mode='circular') # (1,c,h+2,w+2)
        world_share = F.unfold(world_share.float(), kernel_size=(3,3), padding=0).reshape(1,self.c,9,self.h,self.w)

        new_world = (winner_mask * world_share.int()).sum(dim=2) # (1,c,h,w)
        new_world += world_remain

        self.world = new_world[0].int()        
    
        if self.faucet:
            self.world[:,self.h//2-1:self.h//2+1,self.w//2-1:self.h//2+1] += torch.randint_like(self.world[:,self.h//2-1:self.h//2+1,self.w//2-1:self.h//2+1],0,2)
        elif self.tapestry:
            self.world[:,self.h//2,self.w//2] += 1
        

    def reset(self,circle=False):
        """
            Resets the automaton
        """
        if(circle):
            self.circle_mask = torch.where((self.hgrid-self.h//2)**2 + (self.wgrid-self.w//2)**2 < 400, torch.randint_like(self.world,0,5), 0).to(self.device)
            self.world = self.circle_mask
        else:
            self.world = torch.zeros_like(self.world)
    
    def process_event(self, event, camera=None):
        """
        DEL -> reset
        C -> circle
        T -> tapestry
        F -> faucet
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                self.reset()
            if event.key == pygame.K_c :
                self.reset(circle=True)
            if event.key == pygame.K_t :
                self.tapestry = not self.tapestry
                self.faucet = False
            if event.key ==pygame.K_f :
                self.faucet = not self.faucet
                self.tapestry = False