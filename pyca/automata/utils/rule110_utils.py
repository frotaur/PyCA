import numpy as np
import os, sys
import json
from itertools import product, chain
import matplotlib.patches as mpatches
import torch

import matplotlib.pyplot as plt
import matplotlib

"""Rule 110 Simulation with PyTorch"""
def step_torch(c):
    y = c
    x = torch.roll(y, 1) 
    z = torch.roll(y, -1) 

    x_and_y_and_z = x*y*z
    y_or_z = ((y+z)>0).int()
    return torch.logical_xor(x_and_y_and_z , y_or_z).int()

def save_spacetime_torch(c, tim, save_name="sim", device="cuda:0", dpi=500, figsize=50):
    size = c.shape[0]
    M = torch.empty((tim, size), dtype=torch.uint8).to(device)
    for t in range(tim):
        M[t] = c
        c = step_torch(c)

    fig, ax=plt.subplots(figsize=(figsize, figsize))
    ax.matshow(M.cpu(), cmap="binary")
    ax.set_xticks([])
    ax.set_yticks([])
        
    plt.savefig(f'figures/{save_name}.png', format='png', dpi=dpi, bbox_inches="tight")
    return M[-1]

def get_spacetime_torch(c, tim, device="cuda:0", dpi=500, figsize=50):
    size = c.shape[0]
    M = torch.empty((tim, size), dtype=torch.uint8).to(device)
    for t in range(tim):
        M[t] = c
        c = step_torch(c)
    return M


"""Rule 110 simplistic simulation implementation"""
def create_f_dict(rule):
    f_dict={}
    for i in range(1, -1, -1):
        for j in range(1, -1, -1):
            for k in range(1, -1, -1):
                num=2**(4*i+2*j+k)
                if rule>=num:
                    rule=rule-num
                    f_dict[(i,j,k)]=np.uint(1)
                else:
                    f_dict[(i,j,k)]=np.uint(0)
    return f_dict


def sim_cyclic(config, time, l, f_dict, shift=0):
    for t in range(time):
        new_config=np.empty(l, dtype=np.uint)
        for i in range(1, l-1):
            new_config[i+shift]=f_dict[(config[i-1],config[i],config[i+1])]
        new_config[0]=f_dict[(config[(-1-shift)%l],config[(-shift)%l],config[(1-shift)%l])]
        new_config[l-1]=f_dict[(config[(l-2-shift)%l],config[(l-1-shift)%l],config[(l-shift)%l])]
        config=new_config
    return config

def get_space_time(config, tim, grid_size, cell_size, f_dict, skip_between_steps=True):
    M=[config]
    l=grid_size
    for t in range(tim):
        for _ in range(cell_size):
            new_config=np.empty(l, dtype=np.uint)
            for i in range(l):
                new_config[i]=f_dict[(config[(i-1)%l],config[i],config[(i+1)%l])]
            config=new_config
            if not skip_between_steps:
              M.append(config)
        if skip_between_steps:
          M.append(config)
    return M[:-1]


"""CA symmetries tools"""
def symmetry(rule):
    w = [int(x) for x in np.binary_repr(rule, 8)]
    w[1], w[4] = w[4], w[1]
    w[3], w[6] = w[6], w[3]
    total = 0
    #print("symmetric_rule:" , w)
    for index, val in enumerate(reversed(w)):
        total += (val * 2**index)
    return total

def dual(rule):
    w = [int(x) for x in np.binary_repr(rule, 8)]
    w = [1-w[7-i] for i in range(8)]
    total = 0
    for index, val in enumerate(reversed(w)):
        total += (val * 2**index)
    return total

def get_all_symmetries(rule):
    #print("original:      ", rule)
    #print("symmetry:      ", symmetry(rule))
    #print("dual:          ", dual(rule))
    #print("symmetry dual: ", symmetry(dual(rule)))
    return [rule, symmetry(rule), dual(rule), symmetry(dual(rule))]

def get_repre(rule):
    return min([rule, symmetry(rule), dual(rule), symmetry(dual(rule))])


"""Extract gliders"""
ether = [0,0,0,1,0,0,1,1,0,1,1,1,1,1]
str_ether = "".join(str(s) for s in ether)

def extract_phases(config, rule=110, t=1):
    f_dict=create_f_dict(rule)

    config=50*ether+config+50*ether
    for ind in range(t):
      new_config = sim_cyclic(config, 1, len(config), f_dict)
      config = new_config
      str_c = "".join(str(s) for s in new_config)
      i = str_c.index(str_ether)
      str_c = str_c[i:]
      while str_c.startswith(str_ether):
        str_c = str_c[14:]
      i = str_c.rfind(str_ether)
      str_c = str_c[:i]  
      while str_c.endswith(str_ether):
        str_c = str_c[:-14]
    return [int(i) for i in str_c]

def get_glider_period(glider, rule=110):
    f_dict=create_f_dict(rule)
    config=50*ether+glider+50*ether
    new_glider = extract_phases(glider)
    per = 1
    while new_glider != glider:
        new_glider = extract_phases(new_glider)
        per+=1
    return per
    

def crop(config, rule=110):
    f_dict=create_f_dict(rule)
    
    str_c = "".join(str(s) for s in config)
    i = str_c.index(str_ether)
    str_c = str_c[i:]
    while str_c.startswith(str_ether):
        str_c = str_c[14:]
    i = str_c.rfind(str_ether)
    str_c = str_c[:i]
    while str_c.endswith(str_ether):
        str_c = str_c[:-14]
    return [int(i) for i in str_c]


"""Compute distances"""

right_diag_start = "000101100011101011111"
E_hit_diag = {"1001101111110001": 0, "1000100010011011": 0,
             "0000000100110110": 1, 
             "0111110001001100": 2, 
             "1001101111110001": 3, 
             "0000000100110111": 4, 
             "0111110101111101": 5}

RL_hit_diag = {"0000000000000000": 0,
          "0100110001001101": 1,
          "0111110001011111": 2,
          }


def to_str(config):
    return "".join(str(s) for s in config)

def get_first_glider(row):
    i = row.index(str_ether)
    row = row[i:]
    while row.startswith(str_ether):
        row = row[len(str_ether):]
    i = row.index(str_ether)
    row = row[:i]
    return row

def config_diagonal_distance(config, rule=110, hit=E_hit_diag, print_diag=True):
    f_dict=create_f_dict(rule)
    M = [to_str(config)]
    intM = [config]

    for ind in range(300):
        new_config = sim_cyclic(config, 1, len(config), f_dict)
        config = new_config
        M.append(to_str(config))
        intM.append(config)

    #find diagonal start of left glider
    for row_i, row in enumerate(M):
        glider1 = get_first_glider(row)
        if right_diag_start==glider1:
            break
    
    col_j = M[row_i].index(right_diag_start)+7 #(row_i, col_j) is the upper left corner of the first ether triangle of the diagonal start

    #continue the diagonal until we hit the second glider
    triangle = M[row_i][col_j:col_j+4]+M[row_i+1][col_j:col_j+3]+M[row_i+2][col_j:col_j+2]+M[row_i+3][col_j:col_j+1]
    triangle_count=0
    while triangle=="0001001011":
        triangle_count+=1
        
        intM[row_i][col_j]+=2
        intM[row_i][col_j+1]+=2
        intM[row_i][col_j+2]+=2
        
        row_i = row_i+3
        col_j = col_j+2
        triangle = M[row_i][col_j:col_j+4]+M[row_i+1][col_j:col_j+3]+M[row_i+2][col_j:col_j+2]+M[row_i+3][col_j:col_j+1]

    hit_pattern=M[row_i][col_j:col_j+4]+M[row_i+1][col_j:col_j+4]+M[row_i+2][col_j:col_j+4]+M[row_i+3][col_j:col_j+4]

    if print_diag:
        plt.matshow(intM) 
    #    plt.close()

    abs_dist = triangle_count
    rel_dist = hit[hit_pattern]
    
    #print("absolute triangle distance: ", abs_dist)
    #print("relative triangle distance: ", rel_dist)

    return rel_dist, abs_dist

"""this function works for gliders of type E -> E with hit=E_hit_diag
                       for gliders of type E -> RL with hit=RL_hit_diag
    with num_ether being either 6,7,8 or 9"""
def diagonal_distance(e1, e2, num_ethers, rule=110, hit=E_hit_diag, print_diag=True):
    config = 10*ether+e1+num_ethers*ether+e2+10*ether
    return config_diagonal_distance(config, rule=rule, hit=hit, print_diag=print_diag)

right_column_start="000100110111000111011"
E_hit_col={"0011111011": 0, "0111010111": 0, "0000100011": 0,
             "0111111011": 1, "0011100000": 1,
             "0100111100": 2,
             "0000101100": 3, "0100110111": 3,
             "modulus": 4}

RL_hit_col={"0000000010":0, "0110000011":0,
            "0100001110":1, "0111101111":1,
            "modulus": 2,
            }

def config_column_distance(config, rule=110, hit=E_hit_col, print_diag=True):
    f_dict=create_f_dict(rule)
    M = [to_str(config)]
    intM = [config]

    for ind in range(200):
        new_config = sim_cyclic(config, 1, len(config), f_dict)
        config = new_config
        M.append(to_str(config))
        intM.append(config)

    #find diagonal start of left glider
    for row_i, row in enumerate(M[::-1]):
        glider1 = get_first_glider(row)
        if right_column_start==glider1 and row_i>5:
            break

    row_i = len(M)-row_i-1
    col_j = M[row_i].index(right_column_start)+len(right_column_start) #(row_i, col_j) is the upper left corner of the first ether triangle of the diagonal start

    #continue the column until we hit the second glider

    triangle = M[row_i][col_j:col_j+4]+M[row_i+1][col_j:col_j+3]+M[row_i+2][col_j:col_j+2]+M[row_i+3][col_j:col_j+1]
    triangle_count=0
    while triangle=="0001001011":
        triangle_count+=1
        
        intM[row_i][col_j]+=2
        intM[row_i][col_j+1]+=2
        intM[row_i][col_j+2]+=2
        
        row_i = row_i-1
        col_j = col_j+4
        triangle = M[row_i][col_j:col_j+4]+M[row_i+1][col_j:col_j+3]+M[row_i+2][col_j:col_j+2]+M[row_i+3][col_j:col_j+1]

    hit_pattern=M[row_i][col_j:col_j+10]
    if print_diag:
        print(hit_pattern)

    if print_diag:
        plt.matshow(intM)

    abs_dist = triangle_count
    rel_dist = (2*(triangle_count-1)+hit[hit_pattern])%hit["modulus"]

    #print("absolute distance: ", triangle_count)
    #print("relative triangle distance: ", rel_dist)

    return rel_dist

"""this function works for gliders of type E -> E with hit=E_hit
                       for gliders of type E -> RL with hit=RL_hit
    with num_ether being either 6,7,8 or 9"""
def column_distance(e1, e2, num_ethers, rule=110, hit=E_hit_col, print_diag=True):
    config = 6*ether+e1+num_ethers*ether+e2+6*ether
    return config_column_distance(config, rule=rule, hit=hit, print_diag=print_diag)
    
def get_relative_distances(e1, e2, num_ethers, rule=110, hit="E", print_diag=True):
    if hit =="E":
        hit_diag = E_hit_diag
        hit_col  = E_hit_col
    elif hit == "RL":
        hit_diag = RL_hit_diag
        hit_col = RL_hit_col
    col_dist = column_distance(e1, e2, num_ethers, rule, hit=hit_col, print_diag=print_diag)
    diag_dist, abs_diag_dist= diagonal_distance(e1, e2, num_ethers, rule, hit=hit_diag, print_diag=print_diag)
    #print("relative col dist: ", col_dist)
    #print("relative diag dist: ", diag_dist)
    #print("absolute diag dist: ", abs_diag_dist)
    return col_dist, diag_dist, abs_diag_dist

def get_config_relative_distances(config, rule=110, hit="E", print_diag=True):
    if hit =="E":
        hit_diag = E_hit_diag
        hit_col  = E_hit_col
    elif hit == "RL":
        hit_diag = RL_hit_diag
        hit_col = RL_hit_col
    col_dist = config_column_distance(config, rule, hit=hit_col, print_diag=print_diag)
    diag_dist, abs_diag_dist= config_diagonal_distance(config, rule, hit=hit_diag, print_diag=print_diag)
    #print("relative col dist: ", col_dist)
    #print("relative diag dist: ", diag_dist)
    #print("absolute diag dist: ", abs_diag_dist)
    return col_dist, diag_dist, abs_diag_dist

def create_dict_with_basic_gliders(path="data/gliders.json"):
    """Creates a dictionary with all the simple gliders found in rule 110 dynamics.
    Each glider is present in all its phases, and is thus represented as a list"""
    D = {}

    ether = [0,0,0,1,0,0,1,1,0,1,1,1,1,1]
    D["ether"] = ether

    # A gliders
    A_0 =    [0,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,1,1,1,1]
    A = [A_0]+[extract_phases(A_0, t=i+1) for i in range(3)]
    D["A"] = A

    A2_0 =   [0,0,0,1,0,0,1,1,0,1,1,1]
    A2 = [A2_0]+[extract_phases(A2_0, t=i+1) for i in range(3)]
    D["A2"] = A2

    A3_0 =   [0,0,0,1,1,1,0,1,0,0,1,1,0,1,1,1,1,1]
    A3 = [A3_0]+[extract_phases(A3_0, t=i+1) for i in range(3)]
    D["A3"] = A3

    A4_0 =   [0,0,0,1,1,1,0,1,1,1]
    A4 = [A4_0]+[extract_phases(A4_0, t=i+1) for i in range(3)]
    D["A4"] = A4


    #B gliders
    B  = [0,0,0,1,1,1,1,1]
    D["B"] = [B]
    B1 = [1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,1]
    D["B1"] = [B1]
    B2 = [0,0,0,1,0,1,1,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1]
    D["B2"] = [B2]
    B3 = [0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1]
    D["B3"] = [B3]

    #C gliders
    C1 = [0,0,0,1,0,0,0,1,1]
    D["C1"] = [C1]

    C2_0 =   [0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1]
    C2 = [C2_0]+[extract_phases(C2_0, t=i+1) for i in range(6)]
    D["C2"] = C2
    C3 = [0,0,0,1,1,1,0,0,0,1,1]
    D["C3"] = [C3]

    #D gliders
    D1 = [0,1,1,1,0,1,1,1,1,1,1]
    D["D1"] = [D1]
    D2 = [0,0,0,1,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1]
    D["D2"] = [D2]

    #E gliders
    E_0 = [0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1]
    E = [E_0]+[extract_phases(E_0, t=i+1) for i in range(29)]
    D["E"] = E

    E1_0 = [0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,1]
    E1 = [E1_0]+[extract_phases(E1_0, t=i+1) for i in range(29)]
    D["E1"] = E1

    E2_0 = [1,1,1,1,1,0,0,0,1,1,1,1,1]
    E2 = [E2_0]+[extract_phases(E2_0, t=i+1) for i in range(29)]
    D["E2"] = E2

    E3_0 = [0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1]
    E3 = [E3_0]+[extract_phases(E3_0, t=i+1) for i in range(29)]
    D["E3"] = E3

    E4_0 = [0,0,0,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1,1,1,1,1]
    E4 = [E4_0]+[extract_phases(E4_0, t=i+1) for i in range(29)]

    E5_0 = [0,0,0,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1]
    E5 = [E5_0]+[extract_phases(E5_0, t=i+1) for i in range(29)]
    D["E4"] = E4

    #F, G, H gliders and a glider gun
    F = [0,0,0,1,0,0,0,0,1,1,0,0,0,1,1]
    D["F"] = [F]

    G1 = [0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,1,1]
    D["G1"] = [G1]
    G2 = [0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,0,0,1,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,1]
    D["G2"] = [G2]
    G3 = [0,0,0,1,0,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1]
    D["G3"] = [G3]

    H = [0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1]
    D["H"] = [H]

    GG = [0,0,0,1,0,1,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,0,1,1]
    D["GG"] = [GG]


    Y_0 = C2[3] + 3*ether + C2[3] + 3*ether + C2[3] + 2*ether + C2[1]
    Y = [Y_0]+[extract_phases(Y_0, t=i+1) for i in range(7)]
    D["Y"] = Y

    N_0 = C2[3] + 3*ether+ C2[3]+ ether + C2[6] + 2*ether + C2[4]
    N = [N_0]+[extract_phases(N_0, t=i+1) for i in range(7)]
    D["N"] = N

    L_0 = E[19]+E1[4]+3*ether+E4[0]+ether+E[17]+ether+E[21]+2*ether+E[25]+ether+E[22]+3*ether+E[18]
    L = [L_0]+[extract_phases(L_0, t=i+1) for i in range(29)]
    D["L"] = L

    PC_0 =   E[0]+ether+E[6]+2*ether+E[7]+E[15]+2*ether+E[18]+ether+E[14] #primary component
    PC = [PC_0]+[extract_phases(PC_0, t=i+1) for i in range(29)]
    D["PC"] = PC

    acceptor = A[0]+ether+A[1]+A[2]
    acceptor_1 = A[1]+ether+A[2]+A[0]
    acceptor_2 = A[2]+ether+A[0]+A[1]
    D["acceptor"] = [acceptor, acceptor_1, acceptor_2]

    rejector = A3[2]
    rejector_1 = A3[0]
    rejector_2 = A3[1]
    D["rejector"] = [rejector, rejector_1, rejector_2]

    SC_0 = E[0]+2*ether+E[12]+E[23]+E[1]+2*ether+E[4]+2*ether+E[0] #standard component
    SC   = [SC_0]+[extract_phases(SC_0, t=i+1) for i in range(29)] 
    D["SC"] = SC

    O_0   = A4[2]+27*ether+A4[1]+37*ether+A4[0]+26*ether+A4[2]
    O   = [O_0]+[extract_phases(O_0, t=i+1) for i in range(2)]
    D["O"] = O

    RL_0 =   E5[4]+ether+E2[0]+3*ether+E4[4]+ether+E[21]+ether+E[25]+2*ether+E[29]+E[26]+3*ether+E[22]
    RL   = [RL_0]+[extract_phases(RL_0, t=i+1) for i in range(29)]
    D["RL"] = RL

    SL_0 = E5[4]+ether+E2[0]+5*ether+E4[10]+E[15]+E[22]+2*ether+E[6]
    SL   = [SL_0]+[extract_phases(SL_0, t=i+1) for i in range(29)]
    D["SL"] = SL

    Ymiddle = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
           [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]]
    D["Ymiddle"] = Ymiddle

    Nmiddle = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
           ether+[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    D["Nmiddle"] = Nmiddle

    Nouter =  [[[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
          [[1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]],
          [[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]],
          [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]],
          [[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]],
          [[0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0]],
          [[1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]],
          [[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]]
    D["Nouter"] = Nouter

    Youter =  [[[1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]],
          [[1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1]],
          [[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]],
          [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]],
          [[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1]],
          [[1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]],
          [[0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0]]]

    D["Youter"] = Youter

    json.dump(D, open(path, "w"))

def create_dict_yn(path="rule110/dict_yn.json"):
    """D_yn is a dictionary which tracks the compatibility of two SC gliders (depending on their relative phases)
    SC[i] is a glider both starting and ending with glider E[i], so the compatibility boils down to the E glider
    E[i]+num_ether*ether+E[j] are compatible if their relative column distance is 0 
    and their relative diagonal distance is 2: this ensures that if the first glider E[i] hits an acceptor or rejector right, 
    the following E[j] also hits it right, it also ensures that if E[i] passes through static data, E[j] will too. 
    If on top of it SC[i] and SC[j] have the right absolute diagonal distance they form an encoded static data symbol,
    34 diagonals yields a Y and 42 diagonals yields a N. Thus, this dictionary is used in the following sense: given a leader,
    our goal is to prolong it by a sequence of SCs to form one appendant, the dictionary is used, given the last E[i] of the leader,
    to prolong the sequence by a compatible SC"""
    E_0 = [0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1]
    E = [E_0]+[extract_phases(E_0, t=i+1) for i in range(29)]

    D_yn = {}
    for i in range(30):
        D_yn[i] = {}
        for j in range(30):
            for num_ether in range(5,10):
                c, d, a = get_relative_distances(E[i], E[j], num_ether, rule=110, print_diag=False)
                if c == 0 and d ==2:
                    if a == 34:
                        D_yn[i]["Y"] = (j, num_ether)
                        print(f"Y, first E: {i}, second E: {j}, ethers: {num_ether}")
                    elif a == 42:
                        D_yn[i]["N"] = (j, num_ether)
                        print(f"N, first E: {i}, second E: {j}, ethers: {num_ether}")
                    else:
                        D_yn[i][0] = (j, num_ether)
                        print(f"0, first E: {i}, second E: {j}, ethers: {num_ether}")

    json.dump(D_yn, open(path, "w"))

def  create_dict_rl(path="rule110/dict_rl.json"):
    """D_rl is the raw leader dictionary that appends a new raw leader to an appendant
    the key is index of the last E in the appendant, value is (index of raw leader, num_ethers)
    this dictionary is used for every non-empty appendant"""

    """D_rl = {1: (1, 4), 2: (4, 7), 3: (3, 4), 4: (24, 4), 5: (25, 4), 6: (26, 4), 7: (27, 4), 8: (16, 4), 9:(21, 4), 10: (18, 4),
        11: (19, 4), 12: (2, 4), 13:(3, 4), 14: (22, 4), 15: (23, 4), 16: (6, 4), 17: (7, 4), 18: (26, 4), 19: (9, 4), 20: (10, 4),
        21: (11, 4), 22: (24, 4), 23: (13, 4), 24: (14, 4), 25: (15, 4), 26: (28, 4), 27: (9, 4), 28: (18, 4), 29: (19, 4), 0: (8, 4)}"""

    #from standard component SC phase to RL phase 
    D_rl = {0: (8, 4), 1: (1, 4), 2: (4, 7), 3: (3, 4), 4: (24, 4), 5: (25, 4), 6: (26, 4), 7: (27, 4), 8: (16, 4), 9:(21, 4),
        10: (18, 4), 11: (19, 4), 12: (2, 4), 13:(3, 4), 14: (22, 4), 15: (23, 4), 16: (6, 4), 17: (7, 4), 18: (26, 4), 19: (9, 4),
        20: (10, 4), 21: (11, 4), 22: (24, 4), 23: (13, 4), 24: (14, 4), 25: (15, 4), 26: (28, 4), 27: (9, 4), 28: (18, 4), 29: (19, 4)}

    
    json.dump(D_rl, open(path, "w"))


def create_dict_sl(path1="rule110/dict_sc_sl.json", path2="rule110/dict_sl_sl.json", path3="rule110/dict_sl_rl.json"):
    """D_rl is the raw leader dictionary that appends a new raw leader to an appendant
    the key is index of the last E in the appendant, value is (index of raw leader, num_ethers)
    this dictionary is used for an empty appendant"""

    #from standard component SC phase to SL phase 
    D_sc_sl = {0: (29, 6), 1: (0, 6), 2: (1, 6), 3: (2, 6), 4: (3, 6), 5: (4, 6), 6: (17, 6), 7: (6, 6), 8: (7, 6), 9: (8, 6), 
        10: (9, 6), 11: (10, 6), 12: (11, 6), 13: (4, 6), 14: (13, 6), 15: (14, 6), 16: (15, 6), 17: (16, 6), 18: (17, 6), 19: (18, 6), 
        20: (19, 6), 21: (2, 6), 22: (3, 6), 23: (22, 6), 24: (23, 6), 25: (6, 6), 26: (7, 6), 27: (18, 6), 28: (9, 6), 29: (10, 6)} 
    json.dump(D_sc_sl, open(path1, "w"))

    #from short leader to short leader
    D_sl_sl = {0: (2, 1), 1: (3, 0), 2: (4, 0), 3: (5, 1), 4: (6, 0), 5: (7, 0), 6: (8, 0), 7: (9, 1), 8: (10, 0), 9: (11, 0), 10: (12, 1),
           11: (13, 1), 12: (14, 0), 13: (15, 0), 14: (16, 0), 15: (17, 1), 16: (18, 1), 17: (19, 0), 18: (20, 1), 19: (21, 1), 
           20: (22, 1), 21: (23, 0), 22: (24, 1), 23: (25, 1), 24: (26, 0), 25: (27, 1), 26: (28, 1), 27: (29, 0), 28: (0, 0), 29: (1, 0)}
    json.dump(D_sl_sl, open(path2, "w"))


    D_sl_rl = {0: (29, 4), 1: (0, 3), 2: (1, 3), 3: (2, 4), 4: (3, 3), 5: (4, 3), 6: (5, 4), 7: (6, 4), 8: (7, 3), 9: (8, 3), 
          10: (9, 4), 11: (10, 4), 12: (11, 3), 13: (12, 4), 14: (13, 4), 15: (14, 4), 16: (15, 4), 17: (16, 3), 18: (17, 4), 19: (18, 4), 
          20: (19, 4), 21: (20, 4), 22: (21, 4), 23: (22, 4), 24: (23, 3), 25: (24, 4), 26: (25, 4), 27: (26, 3), 28: (27, 4), 29: (28, 4)}
    json.dump(D_sl_rl, open(path3, "w"))       

def get_gliders_with_collisions(path="data/gliders.json"):
    Y_collisions = []

    device="cuda:0"
    

    """Get Y collisions"""
    tape_seq = "YY"
    cyclic_system = ["YNYYYY"] 
    config=torch.tensor(encoder(tape_seq, cyclic_system), dtype=torch.uint8).to(device)
    M = get_spacetime_torch(config, 5000, device=device)
    for i, row in enumerate(M):
        s = [int(r) for r in row[56:194].int().cpu().numpy()]
        if not s in Y_collisions:
            Y_collisions.append(s)
            print("yes", i)
    gliders["Y_collisions"] = Y_collisions
    for y in Y_collisions:
        print(len(y))
    json.dump(gliders, open(path, "w"))

def encode_appendant(yn_seq, leader_i):
    global D_yn, ether, PC, SC
    assert (len(yn_seq)%6)==0 and len(yn_seq)>1
    config = []
    
    i=leader_i
    for char_i, C in enumerate(yn_seq):
        j, num_ether = D_yn[str(i)]["0"]
        if char_i ==0:
            config+=ether*num_ether+PC[j]
            j=(j+14)%30
        else:
            config+=ether*num_ether+SC[j]
        k, num_ether = D_yn[str(j)][C]
        config+=ether*num_ether+SC[k]
        i = k
    print(yn_seq, k)
    return config, k

def encode_raw_leader(E_i):
    global D_rl, RL
    RL_i, num_ethers = D_rl[str(E_i)]
    config=ether*num_ethers+RL[RL_i]
    return config, (RL_i+22)%30

def encoder(tape_seq, cyclic_system):
    global Y, N, L
    config=3*ether
    for C in tape_seq:
        if C == "Y":
            config+=ether+Y[0]+2*ether
        if C == "N":
            config+=ether+N[0]+3*ether
    #first leader is chosen in phase 2 mod 6, this will be the phase after each appendant whose length is 1 mod 3
    config+=4*ether+L[2]
 
    e_i = 20
    for appendant in cyclic_system:
        a, e_i = encode_appendant(appendant, leader_i=e_i)
        config+=a
        l, e_i = encode_raw_leader(e_i)
        config+=l
    return config

def pad_patterns(L):
    global ether
    max_len = max([len(l) for l in L])
    for i, l in enumerate(L):
        diff = max_len-len(l)
        num_ethers = diff//len(ether)
        residue = diff%len(ether)
        #print(to_str(L[i]))
        #print(to_str(l + num_ethers*ether + ether[:residue]))
        L[i] = l + num_ethers*ether + ether[:residue]
    return L


create_dict_sl()