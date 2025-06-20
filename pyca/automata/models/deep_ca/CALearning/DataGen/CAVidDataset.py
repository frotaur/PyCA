from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import transforms
import os
import torch
from .CADatagen import LifeLikeCA
import random
import torch.nn as nn

class FastGPTDataset(IterableDataset):
    """
        Dataset class for videos of CA, adapts data to me processed by SimGPT.
        Faster than CAVidDataset, but less flexible.Needs to be used with dataloader num_workers=0,
        batch_size=None. Batch_size is decided during dataset definition, or can be changed 
        later by setting 'batch_size'.

        params :
        img_size : (H,W) size of the world
        num_frames : number of frames to predict on
        batch_size : batch size
        allowed_x,allowed_y : If specified, will draw rules only among allowed_x and allowed_y
        one_rule : if None, generates videos from random rules. If tuple of ints,
        dataset will consist only of that one rule.
        epoch_length : how many examples in epoch (for logging purposes)
        frame_pred : number of frames to predict, for the 'label' part of the dataset
        pred_distance : frame distance at which prediction starts.
        noise_prob : flips pixels with noise_prob probability
        blur : whether to blur the input images with constant kernel
        skip_frames : number of frames to skip between each frame
        wait_time : number of buffer frames to wait before starting video 'recording'
        majority : whether to do a pass that replaces videos with a 'majority' filter. 
        Only work correctly for 2-state CAs.
        vary_init_size : Whether to vary the size of the random noise initial condition
        backwards : If true, reverses the video
        gen_device : device to generate data on <may be deprecated later>
    """
    def __init__(self,img_size, num_frames, batch_size,one_rule=None, 
                 allowed_x : list=None,allowed_y:list=None,epoch_length = 10000,
                 frame_pred:int=1,pred_distance:int=1, noise_prob:float=None, 
                 blur:bool=False,skip_frames:int=1,wait_time : int = 0,
                 majority = False, vary_init_size : bool = False, backwards : bool = False,
                 gen_device='cpu'):
        super().__init__()
        self.debug=False
        self.CAvidgen = LifeLikeCA(img_size)
        self.time_in = num_frames*skip_frames
        self.needed_extra = frame_pred+pred_distance-1
        self.img_size = img_size

        self.fd = pred_distance
        self.fp = frame_pred
        self.noise_prob = noise_prob
        self.blur=blur

        self.blurrer = nn.Conv2d(1,1,3,stride=1,bias=False, padding_mode='circular')
        self.blurrer.weight.data = torch.tensor([[[[1,1,1],[1,1,1],[1,1,1]]]],device=gen_device,dtype=torch.float) # Box blur
        for p in self.blurrer.parameters():
            p.requires_grad = False
        
        self.skip_frames = skip_frames
        self.wait_time = wait_time
        self.vary_init = vary_init_size
        self.backwards = backwards

        self._epoch_length = epoch_length
        self.gen_device = gen_device # I may deprecate this later
        self._batch_size = batch_size

        if(allowed_x is not None and allowed_y is not None):
            self.al_x = torch.tensor(allowed_x).to(gen_device)
            self.al_x_sh = self.al_x.shape[0]
            self.al_y = torch.tensor(allowed_y).to(gen_device)
            self.al_y_sh = self.al_y.shape[0]
        else :
            self.al_x = torch.arange(0,512,1, device=gen_device) # ALl possible indices
            self.al_x_sh = self.al_x.shape[0]

            self.al_y = torch.arange(0,512,1, device=gen_device) # All possible indices
            self.al_y_sh = self.al_y.shape[0]

        self._one_rule=one_rule
        self.majority = majority

    def make_majority(self,x):
        """
            Apply a majority filter to the dataset. Only works for 2-state CAs.

            Args :
            x : (B,T,C,H,W) tensor
        """
        B,T,C,H,W = x.shape
        x = self.blurrer(x.reshape(B*T,C,H,W)) # (B,T,C,H,W), summed
        x = torch.where(x>0.5,1.,0.)

        return x.reshape(B,T,C,H,W)


    def __len__(self):
        return self.epoch_length//self.batch_size

    @property
    def num_frames(self):
        return self.time_in//self.skip_frames
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self,val):
        self._batch_size = val

    @property
    def epoch_length(self):
        return self._epoch_length

    @epoch_length.setter
    def epoch_length(self,val):
        self._epoch_length = val
    
    @property
    def one_rule(self):
        return self._one_rule
    
    @one_rule.setter
    def one_rule(self,val):
        self._one_rule = val


    def get_batch(self, return_rule=False, raw=False):
        """
            args :
            return_rule : in true, returns also an array of the used rules.
            raw : if true, return the raw video, without any extra processing (no wait_time, no cutting, no noise, no blurring, no skipping)

            returns :
            tuple of batched videos, ( (B,T,1,H,W) , (B,T,fp,1,H,W) )
            if return_rule, ( (B,T,1,H,W) , (B,T,fp,1,H,W), array of B 2-uples )
        """
        # ignore index since infinite data
        if(self.vary_init):
            init_mat = self.CAvidgen.get_init_mat_varied(batch_size=self.batch_size,portion_range=(0.5,1))
        else :
            init_mat = self.CAvidgen.get_init_mat(1.,batch_size=self.batch_size) # (B,H,W) For now, do full random

        x_ind = torch.randint(0,self.al_x_sh,(self.batch_size,),device=self.gen_device) # batch of x_indices
        y_ind = torch.randint(0,self.al_x_sh,(self.batch_size,),device=self.gen_device) # batch of y_indices

        vid = self.CAvidgen.evolve(self.time_in+self.needed_extra+self.wait_time,self.al_x[x_ind],self.al_y[y_ind],init_mat=init_mat,
            device=self.gen_device)[:,:,None].to(dtype=torch.float) # (B,T,1,H,W)
        
        if(raw):
            return vid,None
        
        data, answer = self.process_video(vid)

        if(not return_rule):
            if(self.debug):
                return data, answer, vid # separate into data and 'future'
            else:
                return data, answer
        else :
            return data, answer, zip(self.al_x[x_ind].cpu().numpy(),self.al_y[y_ind].cpu().numpy())   
    
    def _sliding_window(self,tensor):
        """
            Given a tensor of shape (B,T_,C,H,W), it builds a tensor which for
            each time T, contains the next frame_pred frames, in a new channel.
            T_ must be equal to T_-frame_pred+1
        """
        B, T_, C, H, W = tensor.shape
        return torch.as_strided(tensor, 
                                size=(B, T_ - self.frame_pred + 1, self.frame_pred, C, H, W), 
                                stride=(C*H*W*T_, C*H*W, C*H*W, H*W, W, 1)) # (B,T,frame_pred,C,H,W)

    def process_video(self, vid:torch.Tensor):
        """
            Processes the video according to all the parameters of the dataset.

            Args:
            video : (B,T,1,H,W)
        """
        if(self.backwards):
            vid = vid.flip(1) # reverses the video
        if(self.majority):
            vid = self.make_majority(vid) # Applies majority filter, both for ground truth and data

        vid = vid[:,self.wait_time:]  # cut out the wait_time part
        data = vid[:,:self.time_in]
        future = vid[:,self.time_in:]
        T=data.shape[1]
        # START BY BUILDING THE GROUND TRUTH
        start_fut = max(0,self.fd-T)
        answer = torch.cat([data[:,self.fd:],future[:,start_fut:self.fp+self.fd-1]],dim=1) # (B,T+pred_frame-1,C,H,W)

        if(self.noise_prob is not None):
            data = torch.where(torch.rand_like(data) < self.noise_prob, 1-data, data)
        
        if(self.blur):
            B,T,C,H,W = data.shape
            data = data.reshape(B*T,C,H,W)
            data = self.blurrer(data)
            data = data.reshape(B,T,C,H,W)
            
        if(self.fp==1):
            answer = answer[:,:,None] # (B,T,1,C,H,W)
        else :
            B,T_,C,H,W = vid.shape

            T_ = T+self.fp-1
            answer = torch.as_strided(answer, 
                                size=(B, T, self.fp, C, H, W), 
                                stride=(C*H*W*T_, C*H*W, C*H*W, H*W, W, 1)) # (B,T,frame_pred,C,H,W)

        
        if(self.skip_frames>1):
            data = data[:,::self.skip_frames]
            answer = answer[:,::self.skip_frames]
            # WARNING : if fp>1 with skip_frames>1, we will try to predict frames we have no way of knowing.
            # TODO Ideally we should ask for fp = fp*skip_frames, and then do the skipping also on frame_pred. 
        
        return data, answer
    
    def get_batch_rules(self, n_examples, rules_x : torch.Tensor, rules_y : torch.Tensor, raw=False):
        """
            Get batch of videos according to rules tensor.

            Args :
            n_examples : int, number of examples to produce per given rule
            rules_x, rules_y : (n_rules,) int tensor of rules
            raw : whether to return the ray video.

            returns : 
            if raw is false : tuple of batched videos, ( (n_rules,n_examples,T,1,H,W) , (n_rules,n_examples,T,1,H,W) ), one for each of the specified rules
            if raw is true : (n_rules*n_examples,T,1,H,W) video, processable by self.process_video
        """
        n_rules = rules_x.shape[0]

        if(self.vary_init):
            init_mat = self.CAvidgen.get_init_mat_varied(batch_size=n_examples*n_rules,portion_range=(0.5,1))
        else :
            init_mat = self.CAvidgen.get_init_mat(1.,batch_size=n_examples*n_rules) # (B,H,W) For now, do full random

        rules_x = rules_x[:,None].expand(-1,n_examples).reshape(n_rules*n_examples) # (n_rules,n_ex) => (n_ex*n_rules,)
        rules_y = rules_y[:,None].expand(-1,n_examples).reshape(n_rules*n_examples) # (n_rules,n_ex) => (n_ex*n_rules,)

        vid = self.CAvidgen.evolve(self.time_in+self.needed_extra+self.wait_time,rules_x,rules_y,init_mat=init_mat,
            device=self.gen_device)[:,:,None].to(dtype=torch.float) # (n_ex*n_rules,T,1,H,W)
        
        if(raw):
            return vid, None
        
        data, answer = self.process_video(vid)

        # Reshape the batch structure :
        B,Ttot,fp,C,H,W = answer.shape
        data = data.reshape(n_rules,n_examples,Ttot,C,H,W) # Separate the different rules in different dims
        answer = answer.reshape(n_rules,n_examples,Ttot,fp,C,H,W)

        return data, answer # ( (n_rules,n_examples,T,1,H,W) , (n_rules,n_examples,T,frame_pred,1,H,W) )
    
    def get_batch_rule(self,rule : tuple[int], raw=False):
        """
            Args :
            rule : tuple (x,y), rule for which to produce a batch of videos

            returns :
            tuple of batched videos, ( (B,time_in,1,H,W) , (B,time_out,1,H,W) )
        """
        if(self.vary_init):
            init_mat = self.CAvidgen.get_init_mat_varied(batch_size=self.batch_size,portion_range=(0.5,1))
        else :
            init_mat = self.CAvidgen.get_init_mat(1.,batch_size=self.batch_size) # (B,H,W) For now, do full random

        x = torch.full((self.batch_size,), fill_value=rule[0],device=self.gen_device)
        y = torch.full((self.batch_size,), fill_value=rule[1], device= self.gen_device)
        
        vid = self.CAvidgen.evolve(self.time_in+self.needed_extra+self.wait_time,x,y,init_mat=init_mat,
            device=self.gen_device)[:,:,None].to(dtype=torch.float) # (B,T,1,H,W)
        
        if(raw):
            return vid, None
        
        data, answer = self.process_video(vid)
        if(self.debug):
            return data, answer, vid
        else : 
            return data, answer
        
    def __iter__(self):
        count = 0
        while count < self.epoch_length//self.batch_size:
            if(self.one_rule is None):
                yield self.get_batch()
            else :
                yield self.get_batch_rule(self.one_rule)
            count+=1
        


class CAVidDataset(Dataset):
    """
        Dataset class for videos of CA.
        Generates infinite data on the fly.

        params :
        img_size : size of the world
        time_in : time steps to be fed in
        time_out : time steps to be predicted
        epoch_length : length of an epoch, can be chosen arbitrarily
        since the dataset is infinite. No two data points will be the same,
        regardless of epoch_length.

        Dataset element is tuple ( (time_in,1,H,W) , (time_out,1,H,W) )

    """

    def __init__(self,img_size, time_in, time_out, epoch_length = 10000,gen_device='cpu'):
        super().__init__()
        self.CAvidgen = LifeLikeCA(img_size)
        self.time_in = time_in
        self.time_out = time_out

        self.img_size = img_size

        self.gen_device = gen_device
        self.epoch_length = epoch_length
    
    def _to_unreadable(self,x,y):
        """
            Transform classic notation for x/y (see wikipedia article), to the more convenient notation
            that I designed, explained in evo_step. EX : S23/B3 is game of life, translated to S12/B8 in 
            my notation.
        """
        if(x==''):
            x=0
        else:
            x = sum([2**int(d) for d in x])
        if(y==''):
            y=0
        else :
            y = sum([2**int(d) for d in y])
        
        return x,y
    
    def __len__(self):
        return self.epoch_length

    def get_rule(self, rule_x:str,rule_y:str, portion=1.):
        """
            Return data point for specified rule. rule_x is #survive, rule_y is #birth.

            params :
            rule_x : str, e.g. "0123" -> cell survives with 0,1,2,3, neighbors
            rule_y : str, same as rule_y, but for birth.
        """
        x,y = self._to_unreadable(rule_x,rule_y)

        init_mat = self.CAvidgen.get_init_mat(portion,batch_size=1)
        vid = self.CAvidgen.evolve(self.time_in+self.time_out,torch.tensor([x]),torch.tensor([y]),
                                   init_mat=init_mat,device=self.gen_device).squeeze(0)[:,None].to(torch.float)

        return vid[:self.time_in], vid[self.time_in:]

    def __getitem__(self, index):
        # ignore index since infinite data
        init_mat = self.CAvidgen.get_init_mat(1.) # (1,H,W) For now, do full random
        vid = self.CAvidgen.evolve(self.time_in+self.time_out,torch.randint(0,513,(1,)),torch.randint(0,513,(1,)),init_mat=init_mat,device=self.gen_device).squeeze(0)[:,None].to(dtype=torch.float) # (T,1,H,W)

        return vid[:self.time_in], vid[self.time_in:]


class FastCAVidDataset(IterableDataset):
    """
        Dataset class for videos of CA. Faster than CAVidDataset, but less flexible.
        Needs to be used with dataloader num_workers=0, batch_size=None. Batch_size is 
        decided during dataset definition, or can be changed later by setting 'batch_size'.

        params :
        img_size : (H,W) size of the world
        time_in : time steps to be fed in
        time_out : time steps to be predicted
        batch_size : batch size
        allowed_x,allowed_y : If specified, will draw rules only among allowed_x and allowed_y
        one_rule : if None, generates videos from random rules. If tuple of ints,
        dataset will consist only of that one rule.
        epoch_length : how many examples in epoch (for logging purposes)
        wait_time : number of buffer frames to wait before starting video 'recording'
        vary_init_size : Whether to vary the size of the random noise initial condition
        backwards : If true, reverses the video
        gen_device : device to generate data on <may be deprecated later>
    """
    def __init__(self,img_size, time_in, time_out, batch_size,one_rule=None, 
                 allowed_x : list=None,allowed_y:list=None,epoch_length = 10000,
                 wait_time : int = 0, vary_init_size : bool = False, backwards : bool = False, 
                 gen_device='cpu'):
        super().__init__()
        self.CAvidgen = LifeLikeCA(img_size)
        self.time_in = time_in
        self.time_out = time_out

        self.img_size = img_size

        self.wait_time = wait_time
        self.vary_init = vary_init_size
        self.backwards = backwards

        self._epoch_length = epoch_length
        self.gen_device = gen_device # I may deprecate this later
        self._batch_size = batch_size

        if(allowed_x is not None and allowed_y is not None):
            self.al_x = torch.tensor(allowed_x).to(gen_device)
            self.al_x_sh = self.al_x.shape[0]
            self.al_y = torch.tensor(allowed_y).to(gen_device)
            self.al_y_sh = self.al_y.shape[0]
        else :
            self.al_x = torch.arange(0,512,1, device=gen_device) # ALl possible indices
            self.al_x_sh = self.al_x.shape[0]

            self.al_y = torch.arange(0,512,1, device=gen_device) # All possible indices
            self.al_y_sh = self.al_y.shape[0]

        self._one_rule=one_rule

    def __len__(self):
        return self.epoch_length//self.batch_size

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self,val):
        self._batch_size = val

    @property
    def epoch_length(self):
        return self._epoch_length

    @epoch_length.setter
    def epoch_length(self,val):
        self._epoch_length = val
    
    @property
    def one_rule(self):
        return self._one_rule
    
    @one_rule.setter
    def one_rule(self,val):
        self._one_rule = val


    def get_batch(self, return_rule=False):
        """
            args :
            return_rule : in true, returns also an array of the used rules.

            returns :
            tuple of batched videos, ( (B,time_in,1,H,W) , (B,time_out,1,H,W) )
            if return_rule, ( (B,time_in,1,H,W) , (B,time_out,1,H,W), array of B 2-uples )
        """
        # ignore index since infinite data
        if(self.vary_init):
            init_mat = self.CAvidgen.get_init_mat_varied(batch_size=self.batch_size,portion_range=(0.5,1))
        else :
            init_mat = self.CAvidgen.get_init_mat(1.,batch_size=self.batch_size) # (B,H,W) For now, do full random
        x_ind = torch.randint(0,self.al_x_sh,(self.batch_size,),device=self.gen_device) # batch of x_indices
        y_ind = torch.randint(0,self.al_x_sh,(self.batch_size,),device=self.gen_device) # batch of y_indices
        vid = self.CAvidgen.evolve(self.time_in+self.time_out+self.wait_time,self.al_x[x_ind],self.al_y[y_ind],init_mat=init_mat,
            device=self.gen_device)[:,:,None].to(dtype=torch.float) # (B,T,1,H,W)
        
        vid = vid[:,self.wait_time:]  # cut out the wait_time part
        if(self.backwards):
            vid = vid.flip(1) # reverses the video
            

        if(not return_rule):
            return vid[:,:self.time_in], vid[:,self.time_in:] # separate into data and 'future'
        else :
            return vid[:,:self.time_in], vid[:,self.time_in:], zip(self.al_x[x_ind].cpu().numpy(),self.al_y[y_ind].cpu().numpy())   

    def get_batch_rules(self, n_examples, rules_x : torch.Tensor, rules_y : torch.Tensor):
        """
            Get batch of videos according to rules tensor.

            Args :
            n_examples : int, number of examples to produce per given rule
            rules_x, rules_y : (n_rules,) int tensor of rules

            returns : tuple of batched videos, ( (n_rules,n_examples,time_in,1,H,W) , (n_rules,n_examples,time_out,1,H,W) ), one for each of the specified rules
        """
        n_rules = rules_x.shape[0]

        if(self.vary_init):
            init_mat = self.CAvidgen.get_init_mat_varied(batch_size=n_examples*n_rules,portion_range=(0.5,1))
        else :
            init_mat = self.CAvidgen.get_init_mat(1.,batch_size=n_examples*n_rules) # (B,H,W) For now, do full random

        rules_x = rules_x[:,None].expand(-1,n_examples).reshape(n_rules*n_examples) # (n_rules,n_ex) => (n_ex*n_rules,)
        rules_y = rules_y[:,None].expand(-1,n_examples).reshape(n_rules*n_examples) # (n_rules,n_ex) => (n_ex*n_rules,)

        vid = self.CAvidgen.evolve(self.time_in+self.time_out+self.wait_time,rules_x,rules_y,init_mat=init_mat,
            device=self.gen_device)[:,:,None].to(dtype=torch.float) # (n_ex*n_rules,T,1,H,W)
        
        vid = vid[:,self.wait_time:]  # cut out the wait_time part
        if(self.backwards):
            vid = vid.flip(1) # reverses the video
        # Reshape the batch structure :
        B,Ttot,C,H,W = vid.shape
        vid = vid.reshape(n_rules,n_examples,Ttot,C,H,W) # Separate the different rules in different dims

        return vid[:,:,:self.time_in], vid[:,:,self.time_in:] # ( (n_rules,n_examples,time_in,1,H,W) , (n_rules,n_examples,time_out,1,H,W) )
    
    def get_batch_rule(self,rule : tuple[int]):
        """
            Args :
            rule : tuple (x,y), rule for which to produce a batch of videos

            returns :
            tuple of batched videos, ( (B,time_in,1,H,W) , (B,time_out,1,H,W) )
        """
        if(self.vary_init):
            init_mat = self.CAvidgen.get_init_mat_varied(batch_size=self.batch_size,portion_range=(0.5,1))
        else :
            init_mat = self.CAvidgen.get_init_mat(1.,batch_size=self.batch_size) # (B,H,W) For now, do full random

        x = torch.full((self.batch_size,), fill_value=rule[0],device=self.gen_device)
        y = torch.full((self.batch_size,), fill_value=rule[1], device= self.gen_device)
        
        vid = self.CAvidgen.evolve(self.time_in+self.time_out+self.wait_time,x,y,init_mat=init_mat,
            device=self.gen_device)[:,:,None].to(dtype=torch.float) # (B,T,1,H,W)
        
        vid = vid[:,self.wait_time:]  # cut out the wait_time part
        if(self.backwards):
            vid = vid.flip(1) # reverses the video
            
        return vid[:,:self.time_in], vid[:,self.time_in:]
        
    def __iter__(self):
        count = 0
        while count < self.epoch_length//self.batch_size:
            if(self.one_rule is None):
                yield self.get_batch()
            else :
                yield self.get_batch_rule(self.one_rule)
            count+=1
        
