from torchenhanced import *
from .modules import MidGPT
import torch.nn as nn, torch
from ..encoders import flex_encdec
import torch.nn.functional as F
from tqdm import tqdm

class FlexiGPT(ConfigModule):
    """
    FlexiGPT Model. Can use any given image Encoder and Decoder, provided they satisfy some constraints.

    For the encoder and decoder to be compatible, they should have the following 'forward' function :
     - Encoder : forward(x) returns either a tuple (encoded,skips) or a tensor, 'encoded'. 
        - encoded is of shape (B,*latent_dims).
        - skips is a tensor of shape (B,*skip_dims) that will be used for skip connection in the decoder.
    - Decoder : forward(x,skips) return a tensor of shape (B,C,H,W).
        - x is of shape (B,*latent_dims)
        - skips is of shape (B,*skip_dims), and is aligned with the skips from the encoder.
    """

    def __init__(self, in_shape:tuple, encoder:dict, decoder:dict, hid_T=256, trans_hidden=None,
                 n_heads= 4, N_T=4, project_encodings:bool = False, dropout=0., img_skip=True,device='cpu',**kwargs):
        """
            Args:
                in_shape : shape of input (T,C,H,W). Output is of same shape, 
                encoder : dict with 'name' and 'params' keys. 'name should be one of the flex_encdec keys, 
                    and params are the params to be given to the model. Should send (B,C,H,W) to (B,E)
                decoder : dict with 'name' and 'params' keys. 'name should be one of the flex_encdec keys, 
                    and params are the params to be given to the model. Should send (B,C,H,W) to (B,E)
                hid_T : hidden channels of temporal GPT module
                trans_hidden : hidden channels of MLP in temporal GPT module. None : mlp_ratio=4
                n_heads : number of attention heads
                N_T : number of layers of temporal encoder
                project_encodings : if True, projects the encodings to the right size to feed to transformer
                img_skip : if True, prediction of the model is summed to the input image
                dropout : dropout rate of temporal encoder
        """
        super().__init__(device=device)

        T, C, H, W = in_shape 

        self.T_in = T

        self.in_shape = in_shape

        self.enc = flex_encdec[encoder['name']](**encoder['params'], device=device)
        self.dec = flex_encdec[decoder['name']](**decoder['params'], device=device)

        self.frame_pred = 1
        
        self.hid = MidGPT(in_d=hid_T, hid_d=hid_T, frame_pred=1, n_layers=N_T, 
                          n_heads=n_heads, attn_size=T, n_hidden=trans_hidden,
                          dropout=dropout)

        self.img_skip = img_skip

        self.project_enc = project_encodings

        if(project_encodings):
            if(not hasattr(self.enc,'get_lat_shape')):
                print('If `project_encodings` is True, encoder should have a get_lat_shape method.')
            lat_chans = self.enc.get_lat_chans(in_shape[1:])
            self.projector = nn.Linear(lat_chans,hid_T)
            self.unprojector = nn.Linear(hid_T,lat_chans)

    def transformer_forward(self, x):
        """
            Runs the transformer part only, on the input x.
            If project_enc is True, projects the input to the 
            right size before running the transformer, 
            and unprojects it afterwards.

            Args:
            x : (B,T,E) encoded tensor

            Returns :
            x : (B,T,fp,E) encoded predicted tensor
        """
        x = self.project_to_hidden(x) # (B,T,H)
        x = self.hid(x)
        x = self.unproject_from_hidden(x)

        return x

    def enc_forward(self, x):
        """
            Given a video batch, returns the encoded version of it.
            For now, cannot be used 

            Args:
            x : (B,T,C,H,W) tensor

            Returns :
            enc, latent_shape, skips :(encoded (B,T,E) tensor, shape of unbatched latent space, None or skips tensor (B,T,*skip_dims))
        """
        B, T, C, H, W = x.shape

        x = x.reshape(B*T, C, H, W)

        x, skips = self.enc(x) # [ (B*T, *latent_dims), (B*T, *skip_dims) or None ]

    
        latent_shape = x.shape[1:] # tuple containing extra dims

        x = x.reshape(B*T, -1) # (B*T, E)

        _, E = x.shape

        x = x.reshape(B,T,E)
        if(skips is not None):
            skips = skips.reshape(B,T,*skips.shape[1:]) # (B,T,*skip_dims)

        return x, latent_shape, skips 

    def dec_forward(self, x, lat_shape, img_skip=None, skips=None):
        """
            Given encoded batch of frames, returns the decoded version of them.

            Args:
            x : (B,E) tensor
            lat_shape : tuple, unbatched latent shape
            img_skip : (B,C,H,W) tensor
            skips : encoder skips, shape (B,skip_dims)

            Returns :
            x : (B,C,H,W) tensor
        """
        B,E = x.shape


        # Here I do in two steps for clarity, but really it should be one
        x = x.reshape(B,*lat_shape) # (B,  latent_dims)

        if(skips is not None):
            x = self.dec(x,skips=skips) # (B, C, H, W)
        else:
            x = self.dec(x)
        
        _ ,C,H,W = x.shape

        if(self.img_skip):
            x+=img_skip

        return x # (B,C,H,W) tensor
    
    def project_to_hidden(self, x):
        """
            Given a encoded latent, projects it to fit in the transformer
        """
        if(self.project_enc):
            return self.projector(x)
    
    def unproject_from_hidden(self, x):
        """
            Given a hidden latent, unprojects it to fit in the transformer
        """
        if(self.project_enc):
            return self.unprojector(x)

    def prep_skip(self, x):
        """
            Prepares tensor used for skip connection.

            Args:
            x : (B,T,*extra_dims) tensor

            Returns :
            skip : (B*T*fp,*extra_dims) tensor or None
        """
        B, T = x.shape[:2]
        extra_dims = x.shape[2:]

        skip = x.reshape(B*T,*extra_dims) # (B*T, extra_dims)
        skip = skip[:,None] # (B*T, 1, extra_dims)
        return skip.expand(-1,self.frame_pred,*([-1]*len(extra_dims))).reshape(B*T*self.frame_pred,*extra_dims) # (B*T*fp, extra_dims)

    def forward(self, x):
        """
            Runs the model : encode x, run it through the transformer, decode x

            Args:
            x : (B,T,C,H,W) tensor

            Returns :
            x : (B,T,fp,C,H,W) tensor of predictions
        """
        B,T, C, H, W = x.shape
        
        img_skip = self.prep_skip(x) # (B*T*fp, C,H,W) tensor for skip connection
    
        x, lat_shape, skips = self.enc_forward(x) # x: (B,T,E), lat_shape: (*latent_dims), skips: (B,T,*skip_dims) or None
        
        if(skips is not None):
            skips = self.prep_skip(skips) # (B*T*fp,*skip_dims) tensor for skip connection
        
        x = self.transformer_forward(x) # Apply temporal translation network (B,T,fp,H)

        x = x.reshape(B*T*self.frame_pred, -1) # (B*T*fp, E)

        x = self.dec_forward(x, lat_shape, img_skip, skips)

        return x.reshape(B,T,self.frame_pred,C,H,W) # (B,T,fp,C,H,W) tensor of predictions

    def save_enc_dec(self, path):
        """
            Saves the weight/config of the encoder and decoder in the given path.
            Ideally, the path is the same where the model weights are saved, but that's 
            for the trainer to decide. Save format is dict, {'encoder':{'config','state_dict'},
            'decoder':{'config','state_dict'}}.

            Args :
            path : path to save the encoder and decoder (including name of file)
        """
        decoder_dict = {'config':self.dec.config, 'state_dict':self.dec.state_dict()}
        encoder_dict = {'config':self.enc.config, 'state_dict':self.enc.state_dict()}

        full = {'decoder':decoder_dict, 'encoder':encoder_dict}

        torch.save(full, path)
    
    def load_enc_dec(self,path):
        """
            Loads the encoder and decoder from the given path.

            Args:
            path : path of file to load weights from (should be generated by save_enc_dec)
        """
        full = torch.load(path, map_location=self.device)
        # if('config' in full['decoder']):
        #     assert self.dec.config==full['decoder']['config'], "Decoder config does not match, should be {} but is {}".format(self.dec.config,full['decoder']['config'])
        # if('config' in full['encoder']):
        #     assert self.enc.config==full['encoder']['config'], "Encoder config does not match, should be {} but is {}".format(self.enc.config,full['encoder']['config'])

        self.dec.load_state_dict(full['decoder']['state_dict'])
        self.enc.load_state_dict(full['encoder']['state_dict'])
    
    def freeze_enc_dec(self):
        """
            Freezes the encoder and decoder
        """
        for param in self.enc.parameters():
            param.requires_grad = False
        for param in self.dec.parameters():
            param.requires_grad = False

    def unfreeze_enc_dec(self):
        """
            Unfreezes the encoder and decoder
        """
        for param in self.enc.parameters():
            param.requires_grad = True
        for param in self.dec.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate(self, idx, max_new_frames, temperature=1.0, do_sample=False):
        """
            Take a conditioning sequence of frames idx (LongTensor of shape (B,T,C,H,W)) and complete
            the sequence max_new_frames times, feeding the predictions back into the model each time.
            Use with model in inference mode (apply model.eval() first)

            Args :
            idx : (B,T,C,H,W) tensor of context tokens. Mostly, it will be B=1 but can do in parallel also
            max_new_frames : number of frames to generate on top of the conditioning sequence
            temperature : softmax temperature (lower -> more conservative sampling)
            do_sample : if True, use multinomial sampling. Otherwise use greedy decoding

            Returns :
            (B,T,C,H,W) LongTensor of including generated frames.
        """

        for _ in tqdm(range(max_new_frames)):
            idx_next = self.generate_next_token(idx,temperature=temperature,do_sample=do_sample)

            idx = torch.cat((idx, idx_next[:,None]), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_next_token(self,idx,temperature=1.0, do_sample=False):
        """
            Generates the next frame in the sequence, given the conditioning sequence idx.

            Args :
            idx : (B,T,C,H,W) tensor of context tokens. Mostly, it will be B=1 but can do in parallel also
            max_new_frames : number of frames to generate on top of the conditioning sequence
            temperature : softmax temperature (lower -> more conservative sampling)
            do_sample : if True, use multinomial sampling. Otherwise use greedy decoding
            top_k : if set to int > 0, only sample from the top k most probable logits

            Returns :
            next predicted frame : (B,C,H,W) tensor of predicted frame,
        """
        idx_cond = idx if idx.shape[1] <= self.T_in else idx[:, -self.T_in:]
        # forward the model to get the logits for the index in the sequence
        logits = self.forward(idx_cond) # (B,T,fp,C,H,W) tensor of predictions
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, -1] / temperature # (B,C,H,W) tensor of logits for the last frame

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.sigmoid(logits) # (B,C,H,W) tensor of probabilities for the last frame

        # either sample from the distribution or take the most likely element
        if do_sample:
            random_unif = torch.rand_like(probs)
            idx_next = torch.where(random_unif < probs, 1., 0.)
        else:
            idx_next = (probs>0.5).float()
            
        return idx_next # (B,C,H,W) tensor of predicted frame