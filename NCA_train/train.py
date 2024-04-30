import sys, os, pathlib

sys.path.append('..')

from nca_trainer import NCA_Trainer, prepare_img, SamplePool
from Automata.models.NCA import NCAModule
from torchenhanced.util import showTens
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

RUN_NAME = 'betta' # Name of the run, for saving/loading and logging
IMG_PATH = 'images/betta.png' # Relative path to the image to train on
TARGET_SIZE = (40,40) # Size of the target image (will be resized to this)
FRAMES = 64 # Number of frames to evolve the NCA before evaluation
target = prepare_img(IMG_PATH,tarsize=TARGET_SIZE, pad=8)
TRAIN_STEPS = 8000 # Number of steps to train
SAVE_EVERY = 1000 # Save a checkpoint every 'save_every' steps
BACKUP_EVERY = 2000 # Save a backup every 'backup_every' steps
STEP_LOG = 100 # Log the loss and image every 'step_log' steps
BATCH_SIZE = 8 # Batch size
DEVICE = 'cuda' # Device to run the model on


model = NCAModule(n_states=16,n_hidden=128)
optimizer = Adam(model.parameters(), lr=5e-3)
scheduler = MultiStepLR(optimizer, milestones=[2000,5000], gamma=0.5)


# showTens(target)
trainer = NCA_Trainer(model=model, tar_image=target,frame_run=FRAMES, optim=optimizer, 
                      scheduler=scheduler, run_name=RUN_NAME, device=DEVICE,save_loc='runs')

state_path = os.path.join('runs','NCANew','state','{RUN_NAME}.state')
if(os.path.exists(state_path)):
    trainer.load_state(state_path)

trainer.train_steps(TRAIN_STEPS, batch_size=BATCH_SIZE, save_every=SAVE_EVERY, 
                    step_log=STEP_LOG,backup_every=BACKUP_EVERY, norm_grads=False,
                    replace_num=2, corrupt=False, varying_frames=False)