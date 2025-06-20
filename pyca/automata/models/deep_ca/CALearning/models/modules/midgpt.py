from torchenhanced import DevModule
import torch.nn as nn, torch
import torch.nn.functional as F

class MidGPT(DevModule):
    def __init__(self, in_d : int, frame_pred : int, hid_d: int, n_layers: int,n_heads: int, attn_size:int, n_hidden:int=None, dropout=0.):
        """
            MidGPT module. Takes in a tensor of shape (B,T,in_d) and outputs a tensor of shape (B,T,in_d).
            The in tensor is a sequence of encoded frames, flattened to a sequence of vectors.

            args :
            in_d : number of input dimensions
            hid_d : number of hidden dimensions
            n_layers : number of layers
            n_heads : number of attention heads
            attn_size : size of max attention window
            n_hidden : number of hidden dimensions in the MLP layers. 
                If None, defaults to 4 * hid_d
        """
        
        super().__init__()

        if(n_hidden is None):
            n_hidden = 4*hid_d
        
        self.attn_size = attn_size

        self.embedder = nn.Linear(in_d,hid_d)
        self.t_embed = nn.Embedding(attn_size,hid_d)

        self.blocks = nn.ModuleList([GPTBlock(hid_d,n_heads,attn_size,n_hidden=n_hidden,dropout=dropout) for _ in range(n_layers)])

        self.ln_final = nn.LayerNorm(hid_d)

        
        self.debedder = nn.Linear(hid_d,in_d*frame_pred)
        self.frame_pred = frame_pred

    def forward(self,x):
        B,T,D = x.shape

        assert T <= self.attn_size, f"Cannot forward sequence of length {T}, attn size is only {self.attn_size}"

        t_pos = torch.arange(0,T,1,dtype=torch.long,device=x.device)[None,:].expand((B,-1))

        x = self.embedder(x) + self.t_embed(t_pos) # (B,T,hid_d)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        x = self.debedder(x).reshape(B,T,self.frame_pred,D) # (B,T,D*frame_pred) -> (B,T,frame_pred,D)


        return x

class GPTBlock(nn.Module):
    """
        One block (~layer) of a GPT model.

        Parameters :

    """
    def __init__(self, embed_dim,n_heads,attn_size,n_hidden=None,dropout=0.1):
        super().__init__()
        if(n_hidden==None):
            n_hidden=int(4*embed_dim)
    
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MaskedSelfAttention(embed_dim=embed_dim,num_heads=n_heads,attn_size=attn_size,dropout=dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        # Module dict to be able to refer to the individual pieces easily later
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(embed_dim, n_hidden),
            act     = nn.GELU(),
            c_proj  = nn.Linear(n_hidden, embed_dim),
            dropout = nn.Dropout(dropout),
        ))
        m=self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class MaskedSelfAttention(nn.Module) :
    """
        Class implementing MaskedSelfAttention. Based on implementation of minGPT.
        TODO : compare to using nn.MultiheadAttention see which one is faster
        
        Parameters:
        embed_dim : int
        num_heads : int
        dropout : float

        Forward : Input shape should be B,L,D, where L<attn_size.
        Returns masked attention matrix of size B,L,L
    """

    def __init__(self,embed_dim,num_heads,attn_size,dropout):
        super().__init__()
        assert embed_dim%num_heads==0, f"Num_heads should divide embed_dim, but {embed_dim}%{num_heads}!=0"
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear layer that generates the q,k,v tensors
        self.qkv_maker = nn.Linear(embed_dim,3*embed_dim)
        
        # self.softmax = nn.Softmax(dim=3)
        # self.attn_drop = nn.Dropout(dropout)
        self.attn_drop = dropout
        #DEBUG : 
        # self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim,embed_dim)
        self.out_drop = nn.Dropout(dropout)

        # Mask for self-attention
        self.register_buffer("mask", torch.tril(torch.ones(attn_size, attn_size))
                                     .reshape(1, 1, attn_size, attn_size))
        

    
    def forward(self,x : torch.Tensor):
        B,L,D = x.shape

        q,k,v = (self.qkv_maker(x)).split(self.embed_dim,dim=2) # (B,L,D)*3

        # Separate the Heads
        q=q.reshape(B,L,self.num_heads,D//self.num_heads).transpose(1,2) # (B,n_head,L,D')
        k=k.reshape(B,L,self.num_heads,D//self.num_heads).transpose(1,2) # (B,n_head,L,D')
        v=v.reshape(B,L,self.num_heads,D//self.num_heads).transpose(1,2) # (B,n_head,L,D')


        att = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.attn_drop) # (B,n_head,L,D')

        att = att.transpose(1,2).reshape(B,L,D) # Reassemble heads

        att=self.out_drop(self.out_proj(att)) # Project

        return att

