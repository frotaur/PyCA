"""
    Variations on the 1D Cellular Automata.
"""

from ...automaton import Automaton
import json
import colorsys
import random
import pygame
import torch
import numpy as np
from pathlib import Path
from importlib.resources import files

class Rule110Universality(Automaton):
    """
        Universality in rule 110.
    """

    def __init__(self, size, wolfram_num : int, random: bool = True):
        """
            Parameters:
            size : 2-uple (H,W)
                Shape of the CA world
            wolfram_num : int
                Number of the wolfram rule
            random : bool
                If True, the initial tape of the automaton is random. Otherwise, the initial tape is 'YN'
        """
        super().__init__(size)
        # Below here I do what I need to do to initialize the automaton
        self.rule = self.convert_wolfram_num(110) # (8,) tensor, rule[i] is 0 if the i'th neighborhood yields 0, 1 otherwise


        self.max_tape_symbols = 6

        # load dictionaries and glider patterns
        self.load_patterns()
        

        #initialize keyboard step sizes
        self.left_step_size = self.w//20
        self.right_step_size = self.w//20

        #set up world parameters
        self.offset = 100 # margin to circumvent coarse graining missing a pattern cut in half at the border
        self.world = torch.zeros((self.w+2*self.offset),dtype=torch.int) # Vector of W elements
        #self.render_size =  60 if self.h <= 600 else int(60*(self.h//600))
        self.render_size =  210
        self.worlds = torch.ones((self.render_size, self.w+2*self.offset), dtype=torch.int) # Vector of (render_size, W) elements
        self.time = 0 # Current time step, to keep track for plotting the full evolution


        self.left_pressed=False
        self.right_pressed=False

        self.reset(random=False)
        
        self.random = random

        self.can_jump_left = False
        self.can_jump_right = False


        #set up colors
        self.color0 = torch.tensor([1.,1.,1.])
        self.color1 = torch.tensor([0.,0.,0.])
        self.ethercolor1 = torch.tensor([0.96,0.96,0.96])
        self.ncolor0 = torch.tensor([1.,1.,1.])
        self.ncolor1 = torch.tensor([0.549, 0.071, 0.086])
        self.ycolor1 = torch.tensor([0.031, 0.322, 0.125])
        self.colors = torch.stack([self.color0, self.color1, self.ethercolor1, self.ncolor0, self.ncolor1, self.ycolor1])

        #set up text params
        self.text_size = int(self.h/45)
        self.label_color = (230, 230, 230)
        font_path = str(files('pyca.interface.files').joinpath('AldotheApache.ttf'))
        self.font = pygame.font.Font(font_path, size=self.text_size)

        #set up arrows
        self.el_size = self.w//30
        arrow_folder = Path(__file__).parent / 'arrows'
        self.arrows = {}
        for arrow in arrow_folder.glob('*.png'):
            self.arrows[arrow.stem] = pygame.transform.scale(pygame.image.load(arrow.as_posix()), (self.el_size, self.el_size)) 
        self.arrow_height = self.h//5
        self.left_arrow_w_pos = self.w-4*self.el_size
        self.right_arrow_w_pos = self.left_arrow_w_pos+1.5*self.el_size


    def generate_appendnt_6(self):
        appendant = np.random.randint(0,2, 6)          
        if 1 not in appendant:                                         # Make sure there is at least one 1 on the tape
            index = np.random.randint(0, 6)
            appendant[index] = 1
        return "".join(['Y' if i==1 else 'N' for i in appendant])

    def generate_random_cyclic_tag_system(self):
        num_appendants = random.choice([1,2,3,4])
        self.cyclic_tag = [self.generate_appendnt_6() for _ in range(num_appendants)]
        print("Random cyclic tag system is: ", self.cyclic_tag)
        self.init_cyclic_tag_data()

    def init_cyclic_tag_data(self):
        self.max_appendant_len = max(len(a) for a in self.cyclic_tag)
        self.min_appendant_len = min(len(a) for a in self.cyclic_tag)

        self.tot_symbols = "".join(self.cyclic_tag)
        self.total_num_of_symbols = len("".join(self.cyclic_tag))

        self.num_ys = self.tot_symbols.count("Y")
        self.num_ns = self.tot_symbols.count("N")
        self.num_empty = self.cyclic_tag.count("")
        self.num_non_empty = len(self.cyclic_tag)-self.num_empty

        self.long_ossifier_distance =  int((76*self.num_ys+80*self.num_ns+60*self.num_non_empty+43*self.num_empty)//4)*4*12+3    #to be double-checked, is from Cooks paper concrete view of rule 110 computation 

    def load_patterns(self):
        self.dict_yn = json.load(open('pyca/automata/utils/rule110/dict_yn.json', 'r'))
        self.dict_rl = json.load(open('pyca/automata/utils/rule110/dict_rl.json', 'r'))
        self.dict_oss = {0: (1, 1), 1: (2, 0), 2:(0, 0)} #dictionary for ossifiers; example: if last appended ossifier is O[1], then what gets prepended is 0[2]+0*ether+(short or long distance * ether) 
        self.gliders = json.load(open('pyca/automata/utils/rule110/gliders.json', 'r'))

        self.ether = self.gliders['ether']
        self.str_ether = self.to_str(self.ether)
        self.Y = self.gliders['Y']
        self.N = self.gliders['N']
        self.L = self.gliders['L']
        self.C2 = self.gliders['C2']
        self.strC2 = ["".join(str(s) for s in self.ether+c+self.ether) for c in self.gliders['C2']]
        self.PC = self.gliders['PC']
        self.SC = self.gliders['SC']
        self.O = self.gliders['O']
        self.strO = ["".join(str(s) for s in o) for o in self.gliders['O']]
        self.RL = self.gliders['RL']
        self.strE = ["".join(str(s) for s in e) if len(e)> 20 else "".join(str(s) for s in e+self.ether) for e in self.gliders['E']]
        self.Ymiddle = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
           [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]]
        self.Nmiddle = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           self.ether+[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
        self.Nouter =  [[[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
          [[1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]],
          [[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]],
          [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]],
          [[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]],
          [[0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]],
          [[1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]],
          [[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]]
        self.Youter =  [[[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]],
          [[1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1]],
          [[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
          [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]],
          [[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1]],
          [[1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]],
          [[0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0]]]

        self.Y_middle_torch = [torch.tensor(e, dtype=torch.uint8) for e in self.pad_patterns(self.Ymiddle)]
        self.Y_outer_torch = [[torch.tensor(e1, dtype=torch.uint8), torch.tensor(e2, dtype=torch.uint8)] for e1, e2 in self.Youter]
        self.N_middle_torch = [torch.tensor(e, dtype=torch.uint8) for e in self.pad_patterns(self.Nmiddle)]
        self.N_outer_torch = [[torch.tensor(e1, dtype=torch.uint8), torch.tensor(e2, dtype=torch.uint8)] for e1, e2 in self.Nouter]
        self.ether_pattern = torch.tensor(self.ether)

        self.ether_pattern_len = len(self.ether_pattern)  
        self.y_middle_pattern_len = len(self.Y_middle_torch[0])
        self.y_outer_pattern_len = len(self.Y_outer_torch[0][0])
        self.n_middle_pattern_len = len(self.N_middle_torch[0])
        self.n_outer_pattern_len = len(self.N_outer_torch[0][0])

        self.newC2 = [self.ether+c2+self.ether for c2 in self.C2]

    def pad_patterns(self, L):
        max_len = max([len(l) for l in L])
        for i, l in enumerate(L):
            diff = max_len-len(l)
            num_ethers = diff//len(self.ether)
            residue = diff%len(self.ether)
            L[i] = l + num_ethers*self.ether + self.ether[:residue]
        return L

    def to_str(self, config):
        return "".join(str(s) for s in config)

    def encode_raw_leader(self, e_i):
        rl_i, num_ethers = self.dict_rl[str(e_i)]
        config=self.ether*num_ethers+self.RL[rl_i]
        return config, (rl_i+22)%30

    def encode_appendant(self, yn_seq, leader_i):
        assert (len(yn_seq)%6)==0
        config = []
        
        i=leader_i
        for char_i, C in enumerate(yn_seq):
            j, num_ether = self.dict_yn[str(i)]["0"]
            if char_i ==0:
                config+=self.ether*num_ether+self.PC[j]
                j=(j+14)%30
            else:
                config+=self.ether*num_ether+self.SC[j]
            k, num_ether = self.dict_yn[str(j)][C]
            config+=self.ether*num_ether+self.SC[k]
            i = k

        return config, i

    def encode_tape(self):
        config=3*self.ether
        for C in self.init_tape:
            if C == "Y":
                config+=self.ether+self.Y[0]+6*self.ether
            if C == "N":
                config+=self.ether+self.N[0]+7*self.ether
        leader_index=2
        config+=4*self.ether+self.L[leader_index]
        
        return config, (leader_index+18)%30

    def encode_cyclic_tag(self, num_cyclic_tag):
        config=[]
        cyclic_tag = self.cyclic_tag*num_cyclic_tag
        e_i = self.last_e_glider_index
        for appendant in cyclic_tag:
            a, e_i = self.encode_appendant(appendant, leader_i=e_i)
            config+=a+8*self.ether
            l, e_i = self.encode_raw_leader(e_i)
            config+=l
        self.last_e_glider_index = e_i
        return config

    def encode_ossifiers(self, num_ossifiers):
        oss_index, num_ether = self.dict_oss[self.oss_index]
        ossifiers = self.O[oss_index]+self.ether*(num_ether+self.long_ossifier_distance)
        print("new oss index: ", oss_index)
        for o in range(num_ossifiers-1):
            oss_index, num_ether = self.dict_oss[oss_index]
            print("new oss index: ", oss_index)
            ossifiers = self.O[oss_index]+self.ether*(num_ether+self.short_ossifier_distance)+ossifiers
        self.oss_index = oss_index
        return ossifiers

    def crop(self, str_c):
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while str_c.startswith(self.str_ether):
            str_c = str_c[14:]
        i = str_c.rfind(self.str_ether)
        str_c = str_c[:i]
        while str_c.endswith(self.str_ether):
            str_c = str_c[:-14]
        return [int(i) for i in str_c]

    def left_hard_croppable(self, str_c):
        if str_c.startswith(self.str_ether):
            return True
        for e in self.strE:
            if str_c.startswith(e):
                return True

    def left_hard_crop_index(self, str_c):
        if str_c.startswith(self.str_ether):
            return len(self.str_ether)
        for e in self.strE:
            if str_c.startswith(e):
                print("Had to crop garbage")
                return len(e)

    def starts_with_ossifier(self, str_c):
        for i, o in enumerate(self.strO):
            if str_c.startswith(o):
                self.oss_index = i
                #print("updating oss index: ", i)
                return True
        return False

    def left_crop(self, str_c):
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while str_c.startswith(self.str_ether):
            str_c = str_c[14:]
        return [int(i) for i in str_c]

    def left_hard_crop(self, str_c):
        """sometimes, cropping ether to clean up a config is not enough because garbage gliders propagate to the left and make a mess, so this funciton gets rid of them"""
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while self.left_hard_croppable(str_c):
            index = self.left_hard_crop_index(str_c)
            str_c = str_c[index:]

        return [int(i) for i in str_c]

    def right_crop(self, str_c):
        i = str_c.rfind(self.str_ether)
        str_c = str_c[:i]
        while str_c.endswith(self.str_ether):
            str_c = str_c[:-14]
        return [int(i) for i in str_c]

    def get_tape_ends(self, str_c):
        L, R = [], []
        for c in self.strC2:
            if c in str_c:
                L.append(str_c.index(c))
                R.append(str_c.rfind(c))
        if L:
            minl = min(L)
            maxr = max(R)
        else:  #currently there are no tape symbols on the tape
            minl = None
            maxr = None
        return minl, maxr

    def get_init_hidden_world(self):
        #computes the closest safe distance of the first ossifier, this depends on the length of the appendants and the first occurrence of Y + some margin for each tabe symbol
        r_tape_seq = self.init_tape[::-1]
        mask = [int(i =="Y") for i in r_tape_seq]
        first_appendant_index = list(np.array(mask)*np.array([int(len(i)>0) for i in (self.cyclic_tag*len(self.init_tape))[:len(self.init_tape)]])).index(1) #finds the first instance of a non-empty appendant hitting a Y to determine the distance of the first batch of ossifiers
        first_ossifier_distance = len(self.init_tape)*28+self.max_appendant_len*150*first_appendant_index+139
        self.short_ossifier_distance = 67


        #encode first six ossifiers
        num_oss = 6
        oss_index = 2
        ossifiers = self.O[oss_index]+self.ether*first_ossifier_distance
        for _ in range(num_oss-1):
            oss_index, num_ether = self.dict_oss[oss_index]
            print("new oss index: ", oss_index)
            ossifiers = self.O[oss_index]+self.ether*(num_ether+self.short_ossifier_distance)+ossifiers

        ossifiers = 3*self.ether+ossifiers


        #encode tape symbols        
        tape_config, e_i = self.encode_tape()
        self.last_e_glider_index = e_i
        self.oss_index = oss_index

        tape_center = len(ossifiers)+len(tape_config)//2
        self.left_tape_end = len(ossifiers)
        self.right_tape_end = len(ossifiers)+len(tape_config)
        
        #encode cyclic tag system
        num_cyclic_tag = int(self.w//(self.total_num_of_symbols*600))+1
        print(f"appending {num_cyclic_tag} of tag systems")
        cyclic_tag_config = self.encode_cyclic_tag(1)
        init_config = 3*self.ether+ossifiers+tape_config+cyclic_tag_config+3*self.ether


        self.hidden_world = torch.tensor(init_config, dtype=torch.int)
        self.world_center =  tape_center
        self.action_window = (len(ossifiers), len(ossifiers)+len(tape_config))
        
    def update_hidden_world(self):
        str_fc = self.to_str(self.hidden_world.cpu().numpy())
        l_raw_final_config = self.left_hard_crop(str_fc)
        if not self.starts_with_ossifier(self.to_str(l_raw_final_config)): 
            print("skipping update ", self.time)
            return #the leftmost ossifier is just in the middle of a collision, better to skip this update and wait for the ossifier to stabilize
        
        
        i = len(str_fc)-len(l_raw_final_config)
        r_raw_final_config = self.right_crop  (str_fc)
        j = len(r_raw_final_config)

        l, r = self.get_tape_ends(str_fc)
        if not l:
            l = self.world_center
            r = self.world_center

        l = l-i
     
        cropped_center_index = self.world_center-i
        cropped_config = self.right_crop(self.to_str(l_raw_final_config))

        if l < int(np.round(2.5*self.w)):
            #prolonging config to the left
            print("updating left ", self.time)
            ossifiers = self.encode_ossifiers(self.min_appendant_len)
            cropped_config = ossifiers+cropped_config
            cropped_center_index+=len(ossifiers)
            
            
        if j-r < int(np.round(2.5*self.w)):
            # prolonging config to the right
            print("updating right ", self.time)
            cyclic_tag_config = self.encode_cyclic_tag(1)
            cropped_config+=cyclic_tag_config

        cropped_config = int(self.h//7)*self.ether+cropped_config+int(self.h//7)*self.ether

        self.world_center = cropped_center_index+len(int(self.h//7)*self.ether)
        self.hidden_world = torch.tensor(cropped_config, dtype=torch.int)

    def update_tape_position(self):
        str_fc = self.to_str(self.hidden_world.cpu().numpy())

        left_window_edge = self.world_center-self.w//2
        right_window_edge = self.world_center+self.w//2

        left_side = str_fc[:left_window_edge]
        right_side = str_fc[right_window_edge:]

        _, r = self.get_tape_ends(left_side)
        l, _ = self.get_tape_ends(right_side)

        if r:
            self.can_jump_left = True
            self.num_jumps_left = self.world_center-r-self.w//5
        else:
            self.can_jump_left = False
            self.num_jumps_left = 0
        if l:
            self.can_jump_right = True
            self.num_jumps_right = l+self.w//2-self.w//5
        else:
            self.can_jump_right = False
            self.num_jumps_right = 0


    @property
    def worldsurface(self):
        """
            Converts self.worldmap to a pygame surface.

            Can be overriden for more complex drawing operations, 
            such as blitting sprites.
        """
        state_view = pygame.surfarray.make_surface(self.worldmap)
        
        if self.can_jump_left:
            state_view.blit(self.arrows['left_arrow'], (self.left_arrow_w_pos, self.arrow_height))
        if self.can_jump_right:
            state_view.blit(self.arrows['right_arrow'], (self.right_arrow_w_pos, self.arrow_height))

        
        # Write cyclic tag 
        label_surface = self.font.render(f'Cyclic tag: {", ".join(self.cyclic_tag)}', True, self.label_color)
        label_rect = label_surface.get_rect(bottomright=(self.w-10, 0+self.text_size))

        padding = 5
        background_rect = pygame.Rect(
            label_rect.x - padding,
            label_rect.y - padding,
            label_rect.width + (padding * 2),
            label_rect.height + (padding * 2)
        )
        pygame.draw.rect(state_view, (0, 0, 0), background_rect)
        state_view.blit(label_surface, label_rect)

        label_surface = self.font.render(f'Cyclic tag: {", ".join(self.cyclic_tag)}', True, self.label_color)
        label_rect = label_surface.get_rect(bottomright=(self.w-10, 0+self.text_size))

        # Write init tape
        label_surface = self.font.render(f'Init tape: {self.init_tape}', True, self.label_color)
        label_rect = label_surface.get_rect(bottomright=(self.w-10, 0+2*self.text_size))

        background_rect = pygame.Rect(
            label_rect.x - padding,
            label_rect.y - padding,
            label_rect.width + (padding * 2),
            label_rect.height + (padding * 2)
        )
        pygame.draw.rect(state_view, (0, 0, 0), background_rect)
        state_view.blit(label_surface, label_rect)

        return state_view

    def coarse_grain(self, M):
        rows, cols = M.shape
        #print(f"Shape of M: {rows}x{cols}")
        newM = M.clone()

        #Match ether
        windows = M.unfold(1, self.ether_pattern_len, 1)
        matches = (windows == self.ether_pattern.view(1, 1, -1)).all(dim=2)
        # Iterate over the rows with the given step
        for j in range(self.ether_pattern_len):
            newM[:, j:cols-self.ether_pattern_len+j+1] = torch.where(matches & (newM[:, j:cols-self.ether_pattern_len+j+1] == 1), 2, newM[:, j:cols-self.ether_pattern_len+j+1])
        

        #Match outer N components
        Mroll = torch.roll(M, -115, 1)
        windows1 = M.unfold(1, self.n_outer_pattern_len, 1)
        windows2 = Mroll.unfold(1, self.n_outer_pattern_len, 1)
        matches = torch.zeros(windows1.shape[0], windows1.shape[1], dtype=torch.bool)
        for pattern1, pattern2 in self.N_outer_torch:
            matches |= ((windows1 == pattern1.view(1, 1, -1)).all(dim=2) * (windows2 == pattern2.view(1, 1, -1)).all(dim=2))
        # Iterate over the rows with the given step
        offset = 130
        for j in range(self.n_outer_pattern_len+offset):
            newM[:, j:cols-self.n_outer_pattern_len-offset+j+1] = torch.where(matches[:,:-offset] & (newM[:, j:cols-self.n_outer_pattern_len-offset+j+1] <= 1), newM[:, j:cols-self.n_outer_pattern_len-offset+j+1]+3, newM[:, j:cols-self.n_outer_pattern_len-offset+j+1])


        
        #Match middle N components
        windows = M.unfold(1, self.n_middle_pattern_len, 1)
        matches = torch.zeros(windows.shape[0], windows.shape[1], dtype=torch.bool)
        for pattern in self.N_middle_torch:
            matches |= (windows == pattern).all(dim=2)
        # Iterate over the rows with the given step
        l_offset = 50
        r_offset = 40
        offset = l_offset+r_offset
        for j in range(self.n_middle_pattern_len+offset):
            newM[:, j:cols-self.n_middle_pattern_len-offset+j+1] = torch.where(matches[:,l_offset:-r_offset] & (newM[:, j:cols-self.n_middle_pattern_len-offset+j+1] <= 1), newM[:, j:cols-self.n_middle_pattern_len-offset+j+1]+3, newM[:, j:cols-self.n_middle_pattern_len-offset+j+1])
        

        #Match middle Y components
        windows = M.unfold(1, self.y_middle_pattern_len, 1)
        matches = torch.zeros(windows.shape[0], windows.shape[1], dtype=torch.bool)
        for pattern in self.Y_middle_torch:
            matches |= (windows == pattern).all(dim=2)

        matches = matches & (newM[:, :cols-self.y_middle_pattern_len+1]<3)
        # Iterate over the rows with the given step
        l_offset = 50
        r_offset = 40
        offset = l_offset+r_offset
        for j in range(self.y_middle_pattern_len+offset):
            newM[:, j:cols-self.y_middle_pattern_len-offset+j+1] = torch.where(matches[:,l_offset:-r_offset] & (newM[:, j:cols-self.y_middle_pattern_len-offset+j+1] == 1), 5, newM[:, j:cols-self.y_middle_pattern_len-offset+j+1])
        
        
        #Match outer Y components
        Mroll = torch.roll(M, -135, 1)
        windows1 = M.unfold(1, self.y_outer_pattern_len, 1)
        windows2 = Mroll.unfold(1, self.y_outer_pattern_len, 1)
        matches = torch.zeros(windows1.shape[0], windows1.shape[1], dtype=torch.bool)
        for pattern1, pattern2 in self.Y_outer_torch:
            matches |= ((windows1 == pattern1.view(1, 1, -1)).all(dim=2) * (windows2 == pattern2.view(1, 1, -1)).all(dim=2))
        # Iterate over the rows with the given step
        offset = 150
        for j in range(self.y_outer_pattern_len+offset):
            newM[:, j:cols-self.y_outer_pattern_len-offset+j+1] = torch.where(matches[:,:-offset] & (newM[:, j:cols-self.y_outer_pattern_len-offset+j+1] == 1), 5, newM[:, j:cols-self.y_outer_pattern_len-offset+j+1])

        return newM
       
    def process_event(self, event, camera=None):
        """
            SET HEIGHT: 2000, WIDTH: 2500
            N -> new random tape and cyclic tag
            A -> jump left to the next tape symbol
            D -> jump right to the next tape symbol
        """
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_n):
                self.reset(random=self.random) 
            """ 
            if(event.key == pygame.K_n): #Pressed the letter n
                # Picks a random rule
                rule = torch.randint(0,256,(1,)).item()
                self.change_num(rule)
                print('rule : ', rule) """
            if(event.key == pygame.K_c):
                self.color = torch.rand((3,)) # Random color
            if(event.key == pygame.K_LEFT):
                self.world_center -= self.right_step_size
            if(event.key == pygame.K_RIGHT):
                self.world_center += self.right_step_size
            if (event.key == pygame.K_a):
                if self.can_jump_left:
                    self.world_center -= self.num_jumps_left
                    self.can_jump_left = False
            if (event.key == pygame.K_d):
                if self.can_jump_right:
                    self.world_center += self.num_jumps_right
                    self.can_jump_right = False

        self._process_mouse_event(event,camera)

    def _process_mouse_event(self,event,camera):
        """
            Helper function, processes mouse events.
        """
        if event.type == pygame.MOUSEBUTTONDOWN :
            if(event.button == 1):
                self.left_pressed=True
            if(event.button ==3):
                self.right_pressed=True
        if event.type == pygame.MOUSEBUTTONUP:
            if(event.button==1):
                self.left_pressed=False
            elif(event.button==3):
                self.right_pressed=False
        if event.type == pygame.MOUSEMOTION:
            if(self.left_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                # Add interactions when dragging with left-click
        
            elif(self.right_pressed):
                x,y=camera.convert_mouse_pos(pygame.mouse.get_pos())
                # Add interactions when dragging with right-click

    def convert_wolfram_num(self,wolfram_num : int):
        """
            Converts a wolfram number to a rule tensor.
            A tensor, with 8 elements, 0 if the rule is 0, 1 if the rule is 1.
        """
        out = torch.zeros(8,dtype=torch.int8) # Prepare my arary of 8 binary elements
        for i in range(8):
            out[i] = (wolfram_num >> i) & 1
        
        # Now the array out contains the binary representation of wolfram_num

        return out.to(dtype=torch.int) # (Array of 8 elements, where out[i]=0 if the  neighborhood number i yields 0)
    
    
    def reset(self, random = False):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w-1))
        self.time=0

        if(random):
            num_tape_symbols = np.random.randint(1, self.max_tape_symbols) # Pick a random number of tape symbols
            rand_tape = np.random.randint(0,2, num_tape_symbols)           # Pick random tape symbols
            if 1 not in rand_tape:                                         # Make sure there is at least one 1 on the tape
                index = np.random.randint(0, num_tape_symbols)
                rand_tape[index] = 1
            
            self.init_tape = "".join(['Y' if i==1 else 'N' for i in rand_tape])
            self.generate_random_cyclic_tag_system()
            self.get_init_hidden_world()
            self.world = self.hidden_world[self.world_center-int(self.w//2):self.world_center+int(self.w//2)]
        else:
            self.init_tape = 'YN'
            self.cyclic_tag = ["YYYYNY"]
            self.init_cyclic_tag_data()
            #self.generate_random_cyclic_tag_system()
            self.get_init_hidden_world()
            self.world = self.hidden_world[self.world_center-int(self.w//2):self.world_center+int(self.w//2)]
     
    def draw(self):
        # Draw should be called each step
        # We update the _worldmap tensor with the current state of the automaton
        # self._worldmap is a (3,H,W) tensor, which should be seen as an RGB image.
        # self.world is a (W,) array, but we want to set a (3,W) array, so we should duplicate the wrold.
        for i in range(self.render_size):
            self._worldmap[:, (self.time-(self.render_size-i)) % self.h, :] = self.colors[self.worlds[i, self.offset:-self.offset-1]].T         
            
    def step(self):
        """
            Steps the automaton one timestep, recording the state of the world in self.world.
        """



        # One way to do it (which is not parallized ):

        # How to do it all at once
        for i in range(self.render_size):
            indices = (torch.roll(self.hidden_world,shifts=(1))*4+self.hidden_world*2+torch.roll(self.hidden_world,shifts=(-1))*1) # (W,), compute in parallel all sums

            self.hidden_world=self.rule[indices] # This is the same as [ rule[indices[i]] for i in range(W)]
            self.world = self.hidden_world[self.world_center-int(self.w//2)-self.offset:self.world_center+int(self.w//2)+self.offset]
            self.worlds[i, :] = self.world
            self.time += 1
        self.worlds = self.coarse_grain(self.worlds)
        self.update_tape_position()

        if not self.time%(210*int(self.h//210)):
            self.update_hidden_world()





"""
TODO:
- zjistit, kde je bottleneck a zrychlit program (asi pri kropovani a odsekavani garbage glideru)
- kdyz uz budu trackovat (posledni_symbol_pasky, prvni_symbol_pasky) a muj center world bude mensi nez prvni symbol pasky, tak nabidnout velkou spiku doprava, abych mohla k prvnimu symbolu pasky poskocit
- implementovat short raw leadera, ktery je nutny k repre prazdnych appendantu (pak budu muset zmenit, ze long distance bude mezi kazdymi dvema ossificatory!)
"""


