# Generates a dataset of videos of random outer-holistic cellular automata
import torch, os,h5py, numpy as np
from torchvision.io import write_video
from pathlib import Path
from tqdm import tqdm

class LifeLikeCA :
    def __init__(self,size=(300,300)):
        self.h,self.w = size
        self.size = torch.tensor(size)


    def get_init_mat(self,rand_portion,flip_bw=False,batch_size=1):
        """
            Get initialization matrix for CA

            Params : 
            rand_portion : tensor, portion of the screen filled with noise.
            flip_bw : bool, flips black and white.
            batch_size : int, number of initial conditions to generate.
        """
        batched_size = torch.tensor([batch_size,self.h,self.w])
        randsize = (batched_size[1:]*rand_portion).to(dtype=torch.int16) # size of the random square
        randstarts = (batched_size[1:]*(1-rand_portion)/2).to(dtype=torch.int16) # Where to start the index for the random square

        randsquare = torch.where(torch.randn((batch_size,*randsize.tolist()))>0,1,0) # Creates random square of 0s and 1s

        init_mat = torch.zeros((batch_size,self.h,self.w),dtype=torch.int16)
        init_mat[:,randstarts[0]:randstarts[0]+randsize[0],
        randstarts[1]:randstarts[1]+randsize[1]] = randsquare
        init_mat = init_mat.to(torch.int16)

        if(flip_bw):
            init_mat=1-init_mat

        return init_mat # (B,H,W)

    def get_init_mat_varied(self,batch_size=1,portion_range:tuple=(0.5,1.)):
        """
             initalization matrix (B,H,W), where each example is varied in terms
             of the portion of the screen filled with random noise. Also flips b/w
             randomly.

             Args :
             batch_size : self-explanatory
             portion_range : range in which to generate the portions of noise for the inital condition
        """
            # Convert portions to a tensor
        portions_tensor = torch.rand((batch_size,))*(portion_range[1]-portion_range[0])+portion_range[0]

        # Calculate the size of the square for the entire batch
        square_H = (self.h * portions_tensor).int()
        square_W = (self.w * portions_tensor).int()

        # Calculate starting and ending indices for the height and width for the entire batch
        start_H = ((self.h - square_H) // 2).int()
        end_H = (start_H + square_H).int()

        start_W = ((self.w - square_W) // 2).int()
        end_W = (start_W + square_W).int()

        # Create a coordinate grid for height and width
        rows = torch.arange(self.h).view(1, self.h, 1).expand(batch_size, -1, self.w)
        cols = torch.arange(self.w).view(1, 1, self.w).expand(batch_size, self.h, -1)

        # Create the masks based on computed indices
        mask = ((rows >= start_H[:,None,None]) & (rows < end_H[:,None,None]) & (cols >= start_W[:,None,None]) & (cols < end_W[:,None,None])).float() # 1 if in correct portion

        init_mat = (torch.randint(0,2,size=(batch_size,self.h,self.w))*mask).int()

        flip_bw = (torch.randn(size=(batch_size,))>0)[:,None,None].expand(-1,self.h,self.w)

        init_mat = torch.where(flip_bw,init_mat,1-init_mat)

        return init_mat
    
    def evo_step(self,mat,x:int,y:int):
        """
            Evolves one step, using the 'un-readable' convention for rules x and y. In binary, the presence of 1 in the location d of x(y) means
            that if a live(dead) cell has d live neighbors, then it will survive(be born).

            params :
            mat : (H,W) matrix of initial conditions
            x,y : ints, rules for survival and birth respectively.
        """
        wmat, emat = mat.roll(-1, 0), mat.roll(1, 0) # the second argument is the roll axis
        nmat, smat = mat.roll(-1, 1), mat.roll(1, 1) 
        swmat, semat = wmat.roll(1, 1), emat.roll(1, 1)
        nwmat, nemat = wmat.roll(-1, 1), emat.roll(-1, 1)

        count_mat = wmat + emat + nmat + smat + swmat + semat + nwmat + nemat

        return torch.where(mat==1,self.get_nth_bit(x,count_mat),self.get_nth_bit(y,count_mat)).to(torch.int)

    def batch_evo_step(self,mat,x,y):
        """
            Like evo_step, but works on batched data.
            Allows to evolve a batch of initial conditions.

            params : 
            mat : (B,H,W) batch of initial conditions
            x,y : (B,) or int batch (or single) of rules 
        """
        B,H,W = mat.shape
        x= x[:,None,None].expand(-1,H,W)
        y= y[:,None,None].expand(-1,H,W)

        wmat, emat = mat.roll(-1, 1), mat.roll(1, 1) # the second argument is the roll axis
        nmat, smat = mat.roll(-1, 2), mat.roll(1, 2) 
        swmat, semat = wmat.roll(1, 2), emat.roll(1, 2)
        nwmat, nemat = wmat.roll(-1, 2), emat.roll(-1, 2)
        
        count_mat = wmat + emat + nmat + smat + swmat + semat + nwmat + nemat

        return torch.where(mat==1,self.get_nth_bit(x,count_mat),self.get_nth_bit(y,count_mat)) # (B,H,W) of evolved mat
    
    def to_unreadable(self,x : list[str],y : list[str]):
        """
            Transform classic notation for x/y (see wikipedia article), to the more convenient notation
            that I designed, explained in evo_step. EX : S23/B3 is game of life, translated to S12/B8 in 
            my notation.

            <MAYBE LATER CHANGE TO TREAT RULES AS TUPLES (X,Y) INSTEAD OF SEPARATELY>
            params :
            x,y : list of rules in classic notation.

            returns :
            x,y : tensor of rules in my notation.            
        """
        x_out = []
        y_out = []

        # Iterate over x and y
        for rule in x:
            if(rule==''):
                x_out.append(0)
            else:
                x_out.append(sum([2**int(d) for d in rule]))
        
        for rule in y:
            if(rule==''):
                y_out.append(0)
            else:
                y_out.append(sum([2**int(d) for d in rule]))
        
        return torch.tensor(x_out,dtype=torch.int),torch.tensor(y_out,dtype=torch.int)

    def get_nth_bit(self,num, n):
        """
        Returns the nth bit (counting from the right, starting at 0) of num in binary representation.
            n can be a tensor of integers. Num can ALSO be a tensor
        """

        return (num>>n) & 1

    def evolve(self,n_frames,x,y, init_mat,device='cpu'):
        """
            Evolves from the initial matrix, for n_frames using the specified rule. 

            params:
            n_frames : int
            x,y : (B,) int tensor batch of rules.
            init_mat : (B,H,W) batch of initial conditions.

            Return the batched video tensor [B,T,H,W] of boolean vals
        """
        mat = init_mat.to(device)
        with torch.no_grad():
            B,H,W = mat.shape
            frametensor = torch.zeros((B,n_frames,H,W),dtype=torch.int16,device=device)
            frametensor[:,0]=mat
            for i in range(1,n_frames):
                mat = self.batch_evo_step(mat,x,y)
                frametensor[:,i]=mat
        
        return frametensor.to(torch.bool) #[B,T,H,W]

    @torch.no_grad()
    def evolve_video(self,n_frames,x,y,fileDir,record_start=0,device='cpu',init_percent=0.5):
        """
            Like evolve, but saves video instead of returning tensor.

            params:
            n_frames : int
            x,y : (B,) int tensor batch of rules.
            fileDir : str, directory where to save the video.
        """
        os.makedirs(fileDir,exist_ok=True)
        B = x.shape[0]
        init = self.get_init_mat(rand_portion=init_percent,batch_size=B)
        vid = self.evolve(n_frames,x,y,init)[:,record_start:] # [B,T,H,W]
        vid=(vid[...,None].expand(-1,-1,-1,-1,3).to(torch.uint8)*255).to('cpu')
        print(f'Saving {vid.shape[0]} examples to {fileDir}')
        for b in range(vid.shape[0]):
            write_video(os.path.join(fileDir,f'{x[b]}_{y[b]}.mp4'),video_array=vid[b].to('cpu'),fps=24,options={'codec': 'libx264', 'crf': '0'})




 