from nca_trainer import NCA_Trainer, prepare_img, SamplePool
from pyca.automata.models.nca import NCAModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path

RUN_NAME = 'mushroom' # Name of the run, for saving/loading and logging
IMG_PATH = 'targets/mushroom.png' # Relative path to the image to train on

TARGET_SIZE = (40,40) # Size of the target image (will be resized to this)
FRAMES = 64 # Number of frames to evolve the NCA before evaluation
TRAIN_STEPS = 8000 # Number of steps to train
SAVE_EVERY = 250 # Save a checkpoint every 'save_every' steps
BACKUP_EVERY = 1000 # Save a backup every 'backup_every' steps
STEP_LOG = 100 # Log the loss and image every 'step_log' steps
BATCH_SIZE = 8 # Batch size
DEVICE = 'cuda' # Device to run the model on

MODEL_SAVE_LOC = Path(__file__).parent.parent / 'saved_models'
TRAIN_SAVE_LOC = Path(__file__).parent / 'nca_train_state'
IMG_PATH = Path(__file__).parent / IMG_PATH
target = prepare_img(IMG_PATH,tarsize=TARGET_SIZE, pad=8) # Resizes image and adds padding

# Instatiate the model, optimizer and scheduler
model = NCAModule(n_states=16,n_hidden=128) # You can experiments with different model sizes here
# optimizer = Adam(model.parameters(), lr=2e-3)
optimizer = AdamW(model.parameters(), lr=2e-3)
scheduler = MultiStepLR(optimizer, milestones=[2000,5000], gamma=0.5)


# Instantiate the trainer
trainer = NCA_Trainer(model=model, tar_image=target,frame_run=FRAMES, optim=optimizer, 
                      scheduler=scheduler, run_name=RUN_NAME, device=DEVICE,save_loc=TRAIN_SAVE_LOC,
                      model_save_loc = MODEL_SAVE_LOC)


# Launch the training
trainer.train_steps(TRAIN_STEPS, batch_size=BATCH_SIZE, save_every=SAVE_EVERY, 
                    step_log=STEP_LOG,backup_every=BACKUP_EVERY,
                    replace_num=2, corrupt=True, varying_frames=True, pickup=True)