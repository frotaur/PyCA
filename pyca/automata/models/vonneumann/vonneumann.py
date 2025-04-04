from ...automaton import Automaton
from pathlib import Path
import pygame, os, torch

class VonNeumann(Automaton):
    """
        Von Neumann cellular automaton.
    """
    
    def __init__(self, screen_size, element_size, device='cpu'):
        """
            Initializes VonNeumann cellular automaton. Looks nicer if element_size
            divides screen_size.

            Args:
            screen_size : tuple (H,W)
                Size of the VISUALIZED automaton
            element_size : int
                Size of the cells visualized in the automaton.
                Note then that the actual size in cells is screen_size/element_size.
            device : str
                Device to use for the automaton
            
        """
        actual_world_size = (screen_size[0]//element_size, screen_size[1]//element_size)
        super().__init__(actual_world_size)

        self.device=device
        

        self.p_ordinary, self.p_special, self.p_confluent = .3,.3,.3
        state = self.get_rando_para_state(self.p_ordinary, self.p_special, self.p_confluent,.1)

        
        # Excitations :
        self.excitations = (torch.rand((1,*self.size), device=device)<0.5)

        # state=self.make_state_bench() # UNCOMMENT TO USE BENCHMARK 
        self.set_state(state,self.excitations)

        self._worldmap = torch.zeros((3,self.h,self.w), device=device, dtype=torch.int)

        # Generate Sprite list
        self.el_size = element_size
        sprite_folder = Path(__file__).parent / 'sprites'
        self.sprites = {}
        self.sprite_keys ={1:'xp',2:'xm',3:'yp',4:'ym',5:'sxp',6:'sxm',7:'syp',8:'sym',9:'conf',10:'sens0'}

        for sprite in sprite_folder.glob('*.png'):
            self.sprites[sprite.stem] = pygame.transform.scale(pygame.image.load(sprite.as_posix()), (self.el_size, self.el_size))

        self.screen_size = screen_size

        self.select_bar = True
        self.erasing = False
        self.sel_state = 0
        self.mx,self.my = 0,0
        self.mouse_out=True

        saved_folder = Path(__file__).parent / 'premades'
        self.premades = [torch.load(state) for state in saved_folder.glob('*.pt')]
        self.sel_premade = 0

        self.make_state_bench()

    def inj_excitations(self):
        self.excitations = (torch.rand((1,*self.size), device=self.device)<0.2).to(dtype=torch.uint8)

    def get_rando_para_state(self, p_ordinary=None, p_special=None, p_confluent=None, p_dead=0.,
                             state_size=None, batch_size=1):
        """
            Returns a random batch of states with probabilities given as parameters.

            Parameters :
            p_ordinary : Probability of ordinary transmission, if None, uses self
            p_special : Probability of special transmission, if None, uses self
            p_confluent : Probability of confluent, if None, uses self
            p_dead : Probability of dead cell
            state_size : size of the state. If None, uses self.size
            batch_size : Number of states to generate

            Return :
            state : (B,H,W) tensor of uint8
        """
        p_ordinary = self.p_ordinary if p_ordinary is None else p_ordinary
        p_special = self.p_special if p_special is None else p_special
        p_confluent = self.p_confluent if p_confluent is None else p_confluent
        state_size = self.size if state_size is None else state_size

        p_total = p_ordinary+p_special+p_confluent+p_dead

        p_ordinary /= p_total
        p_special /= p_total
        p_confluent /= p_total
        p_dead /= p_total

        state = torch.zeros((batch_size,*state_size),dtype=torch.int).to(self.device)

        # States random :
        uniform = torch.rand(state.shape).to(self.device)

        ordinary_locs = uniform < p_ordinary
        special_locs = (p_ordinary < uniform) & (uniform < p_ordinary + p_special)
        confluent_locs = (p_ordinary + p_special < uniform) & (uniform < p_ordinary + p_special + p_confluent)

        transmission_dirs = torch.randint_like(state, 0, 4)
        state = torch.where(ordinary_locs, transmission_dirs + 1, 0)
        state += torch.where(special_locs, transmission_dirs + 5, 0)
        state += torch.where(confluent_locs, 9, 0)

        return state

    def set_state(self, state, excitations = None):
        """
        	Fills all the tensor for each species given by the state, which should be a tensor of
            shape (1,H,W), with values between 0 and 9. If provided, sets excitations to the given
            (1,H,W) tensor of boolean/int(0,1) values. Otherwise, 0 excitations are set.
            TODO : provide option for random excitations.

            Args :
            state : (B,H,W) tensor of uint8
            excitations : (B,H,W) tensor of uint8 or booleans
        """
        if(excitations is None):
            excitations = torch.zeros_like(state, dtype=torch.bool).to(self.device)
        else :
            excitations = (excitations.to(torch.bool).to(self.device))
        if(state.shape != excitations.shape):
            print("Skipping; State and excitations should have the same shape")
            return
        if(state.shape[1]> self.h or state.shape[2]>self.w):
            print("Skipping; State is too big")
            return
        # Calculate padding size to center the state
        h_pad = (self.h - state.shape[1]) // 2
        w_pad = (self.w - state.shape[2]) // 2

        # Create padded tensors filled with zeros
        padded_state = torch.zeros((1, self.h, self.w), dtype=state.dtype)
        padded_excit = torch.zeros((1, self.h, self.w), dtype=excitations.dtype)

        # Insert state and excitations in center
        padded_state[0, h_pad:h_pad+state.shape[1], w_pad:w_pad+state.shape[2]] = state
        padded_excit[0, h_pad:h_pad+state.shape[1], w_pad:w_pad+state.shape[2]] = excitations

        # Update state and excitations
        state = padded_state
        excitations = padded_excit

        
        state = state.to(self.device)
        # Ordinary transmissions :
        self.ord_e = torch.where(state==1,True,False).to(torch.bool)
        self.ord_w = torch.where(state==2,True,False).to(torch.bool)
        self.ord_s = torch.where(state==3,True,False).to(torch.bool)
        self.ord_n = torch.where(state==4,True,False).to(torch.bool)
        
        # Special transmission :
        self.spe_e = torch.where(state==5,True,False).to(torch.bool)
        self.spe_w = torch.where(state==6,True,False).to(torch.bool)
        self.spe_s = torch.where(state==7,True,False).to(torch.bool)
        self.spe_n = torch.where(state==8,True,False).to(torch.bool)

        # Confluent :
        self.is_conf = torch.where(state==9,True,False).to(torch.bool)
        self.conf_in = torch.zeros_like(self.is_conf)
        self.conf_out = torch.zeros_like(self.is_conf)

        # Sensitized :
        self.is_sens = torch.where(state==10,True,False).to(torch.bool)
        self.sens_state = torch.where(self.is_sens,1,0).to(torch.uint8)
        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # Killed :
        self.is_killed = torch.zeros_like(state,dtype=torch.bool)

        # Init self.inc_ords and self.inc_spes
        self.compute_is_ord()
        self.compute_is_spe()
        # self.compute_ord_excitations()
        # self.compute_spe_excitations()
        self.compute_is_ground()
        
        self.excitations = excitations.to(self.device)

    def get_state(self):
        """
            Returns a tensor of shape (1,H,W), with values between 0 and 9.
        """
        state = torch.zeros((1,*self.size), device=self.device, dtype=torch.int)

        # Ordinary transmissions :
        state = torch.where(self.ord_e,1,state)
        state = torch.where(self.ord_w,2,state)
        state = torch.where(self.ord_s,3,state)
        state = torch.where(self.ord_n,4,state)
        
        # Special transmission :
        state = torch.where(self.spe_e,5,state)
        state = torch.where(self.spe_w,6,state)
        state = torch.where(self.spe_s,7,state)
        state = torch.where(self.spe_n,8,state)

        # Confluent :
        state = torch.where(self.is_conf,9,state)
        
        # TODO : ALLOW SAVING ALL THE DIFFERENT SENSITIZED STATES
        # Sensitized :
        state = torch.where(self.is_sens,10,state)

        return state

    def random_square(self, sq_size=20):
        """
            Reset state to a random square in the middle.
        """
        state = torch.randint(0,10,(1,*self.size), device=self.device)
        mask = torch.zeros_like(state)
        mask[:,self.h//2-sq_size//2:self.h//2+sq_size//2,self.w//2-sq_size//2:self.w//2+sq_size//2]=1
        # mask[:,self.h//2-35:self.h//2+35,self.w//2-35:self.w//2+35]=torch.randint(6,8,(1,70,70), device=self.device)
        state = state*mask
        # state[:,self.h//2-sq_size//2:self.h//2+sq_size//2,self.w//2-sq_size//2:self.w//2+sq_size//2] = mask[:,self.h//2-sq_size//2:self.h//2+sq_size//2,self.w//2-sq_size//2:self.w//2+sq_size//2]
        
        # Excitations :
        excitations = (torch.rand((1,*self.size), device=self.device)<0.5).to(dtype=torch.uint8)

        # state=self.make_state_bench() # UNCOMMENT TO USE BENCHMARK 
        self.set_state(state,excitations)

        state, exci = VonNeumann.get_sens_benchmark(True)
        self.save_state('bench_true',state[None],exci[None])
        state, exci = VonNeumann.get_sens_benchmark(False)
        self.save_state('bench_false', state[None], exci[None])

    @staticmethod
    def get_sens_benchmark(spe=False):
        """
            Return a portion of state which is the benchmark for the sensitive cells.

            Args :
            spe : If True, uses special arrows

            Returns :
            tuple : (state,excit) where both are (9,5) tensors
        """
        def ord_string_state(state_num, spe):
            """
                Gives the arrow+excitation combination to create specific states
            """
            if(spe):
                addon = 4
            else :
                addon=0
            arrows = torch.tensor([1+addon,1+addon,1+addon,1+addon,1+addon])
            
            if(state_num==1):
                return arrows,torch.tensor([0,0,0,0,1])
            elif(state_num==2):
                return arrows,torch.tensor([0,1,0,0,1])
            elif(state_num==3):
                return arrows,torch.tensor([0,0,1,0,1])
            elif(state_num==4):
                return arrows,torch.tensor([1,0,0,0,1])
            elif(state_num==5):
                return arrows,torch.tensor([0,1,1,0,1])
            elif(state_num==6):
                return arrows,torch.tensor([0,1,0,1,1])
            elif(state_num==7):
                return arrows,torch.tensor([0,0,1,1,1])
            elif(state_num==8):
                return arrows,torch.tensor([0,0,0,1,1])
            elif(state_num==9):
                return arrows,torch.tensor([0,1,1,1,1])
        
        out_state = torch.zeros((9,5))
        out_excit = torch.zeros((9,5))
        for i in range(1,10):
            out_state[i-1,:],out_excit[i-1,:] = ord_string_state(i,spe)
    
        return out_state,out_excit

    def make_state_bench(self, spe=False):
        """
            Replaces the state with the benchmark state
        """
        bench_state, bench_excit = VonNeumann.get_sens_benchmark(spe)
        
        self.set_state(bench_state[None],bench_excit[None])
    
    def compute_is_ord(self):
        self.is_ord = (self.ord_e|self.ord_w|self.ord_s|self.ord_n)
    
    def compute_is_spe(self):
        self.is_spe = (self.spe_e|self.spe_w|self.spe_s|self.spe_n)
    

    def compute_is_killed(self):
        self.compute_is_ord()
        self.compute_is_spe()

        self.is_killed = (self.is_ord&self.inc_spes | self.is_spe&self.inc_ords | self.is_conf&self.inc_spes)

    def compute_is_ground(self):
        self.is_ground = (~self.is_ord)&(~self.is_spe)&(~self.is_conf)&(~self.is_sens)

    def compute_conf(self):
        self.compute_inhibitions()

        conf_act = self.conf_out&self.is_conf
        inc_conf_e = torch.roll(conf_act,shifts=-1,dims=2)
        inc_conf_w = torch.roll(conf_act,shifts=+1,dims=2)
        inc_conf_s = torch.roll(conf_act,shifts=-1,dims=1)
        inc_conf_n = torch.roll(conf_act,shifts=+1,dims=1)
        
        # This looses an excitation on top of a confluent if put manually, but it's acceptable
        self.inc_conf_ords = self.is_ord&(inc_conf_e&(~self.ord_e)|inc_conf_w&(~self.ord_w)|inc_conf_s&(~self.ord_s)|inc_conf_n&(~self.ord_n))
        self.inc_conf_spes = self.is_spe&(inc_conf_e&(~self.spe_e)|inc_conf_w&(~self.spe_w)|inc_conf_s&(~self.spe_s)|inc_conf_n&(~self.spe_n))

        self.conf_out = self.conf_in&self.is_conf
        self.conf_in = (~self.inh)&self.inc_ords&self.is_conf
 
    def compute_sens(self):
        inc_exc = (self.inc_ords|self.inc_spes)

        self.is_sens = torch.where((self.is_ground)&inc_exc,True,self.is_sens)

        self.sens_state = ((self.sens_state << 1)+inc_exc)*self.is_sens # Booleans are casted to ints there !

        self.births = torch.zeros_like(self.is_sens,dtype=torch.uint8)

        # I think this is very inefficient also, not sure how to batch it
        self.births = torch.where((self.sens_state==0b10000),1,self.births) # East
        self.births = torch.where((self.sens_state==0b1001),2,self.births) # West
        self.births = torch.where((self.sens_state==0b1010),3,self.births) # South
        self.births = torch.where((self.sens_state==0b10001),4,self.births) # North

        self.births = torch.where((self.sens_state==0b1011),1+4,self.births) # East Special
        self.births = torch.where((self.sens_state==0b1101),2+4,self.births) # West Special
        self.births = torch.where((self.sens_state==0b1110),3+4,self.births) # South Special
        self.births = torch.where((self.sens_state==0b1100),4+4,self.births) # North Special

        self.births = torch.where((self.sens_state==0b1111),9,self.births) # Confluent

        self.is_sens = torch.where(self.births>0,False,self.is_sens)
        self.sens_state = self.sens_state*self.is_sens # Casts to int
    
    def make_births(self):
        # Extinguish remaining excitations on birthing cells
        self.excitations = torch.where(self.births>0,False,self.excitations) 

        # This cannot be batched I think
        self.ord_e = torch.where(self.births==1,True,self.ord_e)
        self.ord_n = torch.where(self.births==4,True,self.ord_n)
        self.ord_w = torch.where(self.births==2,True,self.ord_w)
        self.ord_s = torch.where(self.births==3,True,self.ord_s)

        self.spe_e = torch.where(self.births==5,True,self.spe_e)
        self.spe_n = torch.where(self.births==8,True,self.spe_n)
        self.spe_w = torch.where(self.births==6,True,self.spe_w)
        self.spe_s = torch.where(self.births==7,True,self.spe_s)

        self.is_conf = torch.where(self.births==9,True,self.is_conf)

        self.births = torch.zeros_like(self.births)
        # Recompute which the 'is' thingies
        self.compute_is_ord()
        self.compute_is_spe()

    def compute_ord_excitations(self):
        # This cannot be batched I think
        inc_ord_e = torch.roll(self.ord_w&self.excitations,shifts=-1,dims=2)
        inc_ord_w = torch.roll(self.ord_e&self.excitations,shifts=1,dims=2)
        inc_ord_s = torch.roll(self.ord_n&self.excitations,shifts=-1,dims=1)
        inc_ord_n = torch.roll(self.ord_s&self.excitations,shifts=1,dims=1)

        self.inc_ords = inc_ord_e&(~self.ord_e)|inc_ord_w&(~self.ord_w)|inc_ord_s&(~self.ord_s)|inc_ord_n&(~self.ord_n)

    def compute_inhibitions(self):
        # This cannot be batched I think
        inc_inh_e = torch.roll(self.ord_w&(~self.excitations),shifts=-1,dims=2)
        inc_inh_w = torch.roll(self.ord_e&(~self.excitations),shifts=1,dims=2)
        inc_inh_s = torch.roll(self.ord_n&(~self.excitations),shifts=-1,dims=1)
        inc_inh_n = torch.roll(self.ord_s&(~self.excitations),shifts=1,dims=1)

        self.inh = (inc_inh_e|inc_inh_w|inc_inh_s|inc_inh_n)

    def compute_spe_excitations(self):
        """
            Computes the excitations of the special particles, and returns the purged incoming excitations.
        """
        # This cannot be batched I think
        inc_spe_e = torch.roll(self.spe_w&self.excitations,shifts=-1,dims=2)
        inc_spe_w = torch.roll(self.spe_e&self.excitations,shifts=1,dims=2)
        inc_spe_s = torch.roll(self.spe_n&self.excitations,shifts=-1,dims=1)
        inc_spe_n = torch.roll(self.spe_s&self.excitations,shifts=1,dims=1)

        self.inc_spes = inc_spe_e&(~self.spe_e)|inc_spe_w&(~self.spe_w)|inc_spe_s&(~self.spe_s)|inc_spe_n&(~self.spe_n)
    
    def _check_log_step(self):
        """
            Debug function
        """
        print('============================================================')
        print('============================================================')
        print('----------------------- STEP -----------------------')
        print('============================================================')
        print('============================================================')
        self.check_no_problem()

        self.compute_ord_excitations()
        self.compute_spe_excitations()
        print('----------------------- compute ord and spe excitations -----------------------')
        self.check_no_problem()

        self.compute_conf()
        print('----------------------- compute conf -----------------------')
        self.check_no_problem()
        self.compute_sens()
        print('----------------------- compute sens -----------------------')
        self.check_no_problem()

        self.excitations = ((self.inc_ords|self.inc_spes+self.inc_conf_ords+self.inc_conf_spes)>0).to(torch.uint8)
        self.excitations = torch.where((self.is_conf)*(1-self.conf_in)==1,0,self.excitations)# Remove spurious exictations on top of deactivated conf_in

        # self.excitations = ((self.inc_ords+self.inc_spes)>0).to(torch.int)
        print('----------------------- compute excitations -----------------------')
        self.check_no_problem()

        self.compute_is_killed()
        print('----------------------- compute is killed -----------------------')
        self.check_no_problem()
        self.kill_dead()
        print('----------------------- kill dead -----------------------')
        self.check_no_problem()

        self.make_births()
        print('----------------------- make births -----------------------')
        self.check_no_problem()

        self.compute_is_ground()
        print('compute is ground -----------------------')
        self.check_no_problem()
    
    def step(self):
        self.compute_ord_excitations()
        self.compute_spe_excitations()

        self.compute_conf()
        self.compute_sens()

        self.excitations = (self.inc_ords|self.inc_spes|self.inc_conf_ords|self.inc_conf_spes)
        # self.excitations = torch.where((self.is_conf)&(~self.conf_in),False,self.excitations)# Remove spurious exictations on top of deactivated conf_in

        self.compute_is_killed()
        self.kill_dead()
        self.make_births()

        self.compute_is_ground()
        self._check_no_problem()

    def save_state(self,  path='.', state=None, excitations=None):
        """
            Saves current state. If state and or excitations are provided, saves these instead.
        """
        if(state is None):
            if(excitations is not None):
                print('Please provide both state and excitations, or none of them')
                return
            else:
                state = self.get_state()
                excitations = self.excitations
        if(excitations is None and state is not None):
            print('Please provide both state and excitations, or none of them')
            return

        os.makedirs(path,exist_ok=True)
        to_save = torch.stack((state,excitations),dim=0) # (2,B,H,W)
        torch.save(to_save,path+'/saved_state.pt')

    def process_event(self, event, camera=None):
        """
            B -> Toggle select bar
            O -> Reset to random squar
            E -> Inject random excitations
            Tab -> Cycle pre-sets
            DELETE -> Reset to empty state
            A -> Save state (not great for now)
            0-9 -> Choose drawing state
            DOWN -> Erase mode
            UP -> Excitation mode
            Left-click-> Insert selected state
            Right-click -> Add excitation, or erase chunk
            
            Please stop the simulation before drawing.
        """
        m = self.get_mouse_state(camera)
        self.mx,self.my = int((m.x)//self.el_size), int((m.y)//self.el_size)
        if(self.my>=self.h or self.mx<0 or self.my<0 or self.mx>=self.w):
            self.mouse_out=True
        else:
            self.mouse_out=False
    
        self.mx = min(max(self.mx,0),self.w-1)
        self.my = min(max(self.my,0),self.h-1)

        if event.type == pygame.KEYDOWN:
            if(event.key ==pygame.K_o):
                self.random_square()
            if(event.key == pygame.K_e):
                self.inj_excitations()
            if(event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE):
                self.is_killed = torch.ones_like(self.is_killed)
                self.kill_dead()
            if(event.key == pygame.K_a):
                statio = self.get_state()
                state_excit = torch.stack((statio,self.excitations),dim=0) # (2,H,W)
                os.makedirs('saved_vonno',exist_ok=True)    
                torch.save(state_excit,'saved_vonno/saved_state.pt')
            if(event.key == pygame.K_DOWN):
                self.erasing= True
            if(event.key == pygame.K_UP):
                self.erasing= False
            if(event.key == pygame.K_b):
                self.select_bar = not self.select_bar
            if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                self.sel_state = int(event.unicode)
            if event.key == pygame.K_TAB:
                self.sel_premade = (self.sel_premade + 1) % len(self.premades)
                self.set_state(self.premades[self.sel_premade][0],self.premades[self.sel_premade][1])
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # scroll up
                self.sel_state = (self.sel_state + 1) % 10
            elif event.y < 0:  # scroll down
                self.sel_state = (self.sel_state - 1) % 10
        if event.type == pygame.MOUSEBUTTONDOWN and not self.mouse_out:
            if m.left:
                status = self.get_state()
                status[:,self.my,self.mx] = self.sel_state
                self.set_state(status,excitations=self.excitations)
            if m.right:
                if(not self.erasing):
                    self.excitations[:,self.my,self.mx] = ~self.excitations[:,self.my,self.mx]

        if event.type ==pygame.MOUSEMOTION and not self.mouse_out:
            if m.right and self.erasing:
                self.is_killed[:,max(self.my-5,0):min(self.my+5,self.h),max(self.mx-5,0):min(self.mx+5,self.w)] = 1
                self.kill_dead()

    def kill_dead(self):
        is_alive = ~self.is_killed
        # Try to batch this operation
        self.ord_e = self.ord_e&is_alive
        self.ord_w = self.ord_w&is_alive
        self.ord_s = self.ord_s&is_alive
        self.ord_n = self.ord_n&is_alive

        self.spe_e = self.spe_e&is_alive
        self.spe_w = self.spe_w&is_alive
        self.spe_s = self.spe_s&is_alive
        self.spe_n = self.spe_n&is_alive

        self.is_conf = self.is_conf&is_alive
        self.conf_in = self.conf_in&self.is_conf
        self.conf_out = self.conf_out&self.is_conf

        self.compute_is_ord()
        self.compute_is_spe()

        self.excitations = self.excitations&is_alive

    def _check_no_problem(self):
        """
            Debug function, prints if there are problems in the state.
        """
        # Sum should not be bigger than one, otherwise we have overlapping states :
        problem_mask = (self.ord_e.int()+self.ord_w.int()+self.ord_s.int()+self.ord_n.int()\
                        +self.spe_e.int()+self.spe_w.int()+self.spe_s.int()+self.spe_n.int()\
                        +self.is_conf.int()+self.is_sens.int())>1
        
        problem_ind = problem_mask.nonzero()
        if(problem_mask.any()):
            print('----------------------- FOUND PROBLEM -----------------------')
            print('ord_e : ',self.ord_e[problem_mask])
          
            print('ord_w :',self.ord_w[problem_mask])
            
            print('ord_s :',self.ord_s[problem_mask])
        
            print('ord_n :',self.ord_n[problem_mask])

            print('spe_e :',self.spe_e[problem_mask])

            print('spe_w :',self.spe_w[problem_mask])

            print('spe_s :',self.spe_s[problem_mask])

            print('spe_n :',self.spe_n[problem_mask])

            print('conf :',self.is_conf[problem_mask])

            print('sens :',self.is_sens[problem_mask])
            raise Exception('Found overlapping states at indices : ',problem_ind)
    
    def draw(self):
        self._worldmap = torch.zeros_like(self._worldmap)
        self._worldmap = torch.where(self.ord_e,1,self._worldmap)
        self._worldmap = torch.where(self.ord_w,2,self._worldmap)
        self._worldmap = torch.where(self.ord_s,3,self._worldmap)
        self._worldmap = torch.where(self.ord_n,4,self._worldmap)
        self._worldmap = torch.where(self.spe_e,5,self._worldmap)
        self._worldmap = torch.where(self.spe_w,6,self._worldmap)
        self._worldmap = torch.where(self.spe_s,7,self._worldmap)
        self._worldmap = torch.where(self.spe_n,8,self._worldmap)
        self._worldmap = torch.where(self.is_conf,9,self._worldmap)
        self._worldmap = torch.where(self.is_sens,10,self._worldmap)
    
    @property
    def worldsurface(self):
        """
            Blits element sprites on the screen, generating the correct pygame surface
        """
        elements, excitations, conf_out = self.worldmap
        W,H,_ = elements.shape
        surf = pygame.Surface((self.el_size*W, self.el_size*H))

        for i in range(W):
            for j in range(H):
                status = elements[i,j,0]
                if(status!=9 and status!=0):
                    surf.blit(self.sprites[self.sprite_keys[status]], (i*self.el_size, j*self.el_size))
                elif(status==9) :
                    if(conf_out[i,j,0]>0):
                        surf.blit(self.sprites['conf01'], (i*self.el_size, j*self.el_size))
                    else:
                        surf.blit(self.sprites['conf0'], (i*self.el_size, j*self.el_size))
                if(excitations[i,j,0]>0):
                    surf.blit(self.sprites['excited'], (i*self.el_size, j*self.el_size))

        if(not self.mouse_out):
            pygame.draw.rect(surf, (125,25,25),(self.mx*self.el_size,self.my*self.el_size,self.el_size,self.el_size),1)

        if(self.select_bar):
            bar_el_size = 10
            bar_height = bar_el_size
            bar_width = 11 * bar_el_size  # 10 sprites
            select_surf = pygame.Surface((bar_width+2, bar_height+2))
            select_surf.fill((0, 0, 0))

            # Draw bar outline
            pygame.draw.rect(select_surf, (0, 0, 255), (0, 0, bar_width+2, bar_height+2), 1)
            # Draw sprites
            for i in range(11):
                if i == 9:
                    select_surf.blit(self.sprites['conf0'], (i*bar_el_size+1, 1))
                elif i == 10 :
                    if(self.erasing):
                        pygame.draw.rect(select_surf, (200, 10, 10), (i*bar_el_size+1, 1, bar_el_size-1, bar_el_size-1))
                    else:
                        pygame.draw.rect(select_surf, (125, 125, 20), (i*bar_el_size+1, 1, bar_el_size-1, bar_el_size-1))
                    pygame.draw.rect(select_surf, (100, 100, 250), (i*bar_el_size+1, 1, bar_el_size-1, bar_el_size-1), 1)
                elif i>0 and i<9:
                    select_surf.blit(self.sprites[self.sprite_keys[i]], (i*bar_el_size+1, 1))
                if i == self.sel_state:  # state was selected with number keys
                    pygame.draw.rect(select_surf, (0, 255, 0), (i*bar_el_size+1, 1, bar_el_size, bar_el_size), 1)

            # Blit select bar onto main surface at top right
            bar_x = surf.get_width() - bar_width
            surf.blit(select_surf, (bar_x-2, 2))


        # Check if surface size matches screen_size and resize if needed
        if (surf.get_height(), surf.get_width()) != self.screen_size:
            padded = pygame.Surface(self.screen_size)  # This creates a black surface by default
            # Calculate centering coordinates
            x_offset = (self.screen_size[1] - surf.get_width()) // 2
            y_offset = (self.screen_size[0] - surf.get_height()) // 2
            # Blit original surface onto center of padded surface
            padded.blit(surf, (0, 0))
            surf = padded
    
        return surf

    @property
    def worldmap(self):
        """
            Overriden to return more info about the state for blitting
        """
        return self._worldmap.permute(2,1,0).cpu().numpy(), self.excitations.int().permute(2,1,0).cpu().numpy(), self.conf_out.int().permute(2,1,0).cpu().numpy()