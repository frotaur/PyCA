import sys, os, pathlib

sys.path.append('..')

from nca_trainer import NCA_Trainer, prepare_img, SamplePool
from Automata.models.NCA import NCAModule
from torchenhanced.util import showTens
from torch.optim import Adam, SGD

target = prepare_img('images/whale.png',tarsize=48)
showTens(target)

model = NCAModule(n_states=16,n_hidden=64)

optimizer = Adam(model.parameters(), lr=2e-3)

trainer = NCA_Trainer(model=model, tar_image=target,frame_run=80, optim=optimizer,run_name='whale', device='cuda',save_loc='runs')

if(os.path.exists('runs\\NCANew\\state\\whale.state')):
    trainer.load_state('runs\\NCANew\\state\\whale.state')
trainer.train_steps(20000, batch_size=48, save_every=1000, step_log=300)