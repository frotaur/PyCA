"""
Helper class to train NCA model
"""

import math
from pyca.automata.models.nca import NCAModule, SamplePool
import torch, os
from tqdm import tqdm
from torchenhanced import Trainer
from showtens import gridify
from torchvision.transforms import transforms
from PIL import Image
import math
import wandb

class NCA_Trainer(Trainer):
    def __init__(
        self,
        model: NCAModule,
        tar_image: torch.Tensor,
        frame_run: int = 64,
        frame_delta: int = 32,
        run_name: str = None,
        optim=None,
        scheduler=None,
        *,
        save_loc='./runs',
        model_save_loc='./saved_models',
        device: str = "cpu",
    ):
        """
        NCA trainer class. Uses pixel of ones seed by default.
        Parameters:
            model : NCAModule to train
            tar_image : (4,h,w) Target image to train the model on
            frame_run : number of frames to evolve the NCA before evaluation
            frame_delta : Random variations max on frame_run. Introducing stochasticity encourage model to stabilize
            run_name : Name of the run, for wandb logging
            optim : Optimizer to use
            scheduler : Learning rate scheduler to use
            save_loc : Directory to save checkpoints
            device : Device to run the model on
        """
        super().__init__(
            model, optim, scheduler, save_loc=save_loc, device=device, run_name=run_name, project_name="NCA"
        )
        self.model = model.to(self.device)  # just for auto-completion
        self.tar_image = tar_image.to(self.device)[None]  # (1,4,H,W) Target image padded with 4 pixels

        self.world_size = (tar_image.shape[1], tar_image.shape[2])  # Add padding to the target image
        self.frame_run = frame_run  # Number of frames to evolve the NCA before evaluation
        self.frames_delta = frame_delta  # Random variations max on frame_run
        seed = torch.zeros(self.model.n_states, *self.world_size)
        seed[:, self.world_size[0] // 2, self.world_size[1] // 2] = 1
        self.pool = SamplePool(seed, return_device=self.device)  # Pool of samples for training

        self.save_loc = os.path.join(save_loc, "NCA", f"{self.run_name}")
        self.model_save_loc = os.path.join(model_save_loc, "NCA", f"{self.run_name}")

    def train_steps(
        self,
        steps,
        batch_size,
        *,
        save_every=50,
        step_log: int = None,
        backup_every=float("inf"),
        corrupt=True,
        num_cor=3,
        replace_num=1,
        varying_frames=True,
        pickup=False
    ):
        """
        Main training function. Trains the model for 'steps' steps.

        Parameters:
            steps : int
                Number of steps to train
            batch_size : int
                Batch size
            save_every : int
                Save a checkpoint every 'save_every' steps
            step_log : int
                Log the loss every 'step_log' steps
            backup_every : int
                Save a backup every 'backup_every' steps
            norm_grads : bool
                Normalize gradients before applying them
            corrupt : bool
                Corrupt the samples before training
            num_cor : int
                Number of corruptions to apply
            replace_num : int
                Number of samples to replace by seed. Use replace_num=batch_size to emulate NO POOL
            varying_frames : bool
                Use varying number of frames to evolve the NCA
            Pickup : bool
                If True, will pick up the training from the last checkpoint (if it exists)
        """
        if(pickup):
            self._load_if_exists() # Load the training state if it already exists

        self._init_logger() # Initialize wandb logger
        self.model.train()

        self.step_log = step_log
        self.step_loss = []

        steps_completed = False

        numsteps = 0

        load_bar = tqdm(total=steps, desc="Training", unit="step")
        do_corrupt = False # Start out with no corruption

        while not steps_completed:
            self.do_step_log = numsteps % self.step_log == 0 if self.step_log is not None else False # Log every step_log steps

            if corrupt and self.steps_done > 1000: # Initiate corruptiong after 1000 steps
                do_corrupt = True

            batch, indices = self.pool.sample(
                num_samples=batch_size, replace_num=replace_num, corrupt=do_corrupt, num_cor=num_cor
            )  # get the next batch

            if varying_frames: #  Choose the number of evolution frames
                rand_evo = torch.randint(self.frame_run, self.frame_run + self.frames_delta, (1,)).item()
            else:
                rand_evo = self.frame_run

            state = torch.clone(batch) # TODO : check if necessary
            state = self.model(state, n_steps=rand_evo)  # (B,C,H,W) evolved state

            loss = self.model.loss(state, self.tar_image.expand(batch_size, -1, -1, -1)).mean(
                dim=(1, 2, 3)
            )  # (B,) loss per sample
            loss.mean().backward()


            # Clip the gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)


            self.optim.step()
            self.optim.zero_grad()

            self.log_progress(loss, batch, state)

            self.pool.update(indices, state, batchloss=loss.detach())

            self.scheduler.step()
            # Update the tracked quantities
            self.steps_done += 1
            self.batches += 1
            self.epochs += 1 / self.pool.p_size
            load_bar.update(1)

            numsteps += 1
            self._save_and_backup(save_every=save_every, backup_every=backup_every)

            if numsteps > steps:
                steps_completed = True

    def to_rgb(self, state: torch.Tensor, bg_color: float = 0):
        """
        Convert a state tensor to an RGB tensor
        """
        argb = torch.clamp(self.model.state_to_argb(state), min=0, max=1)  # (B,4,H,W) ARGB tensor
        alpha = argb[:, 3:]  # (B,1,H,W) Alpha channel
        return argb[:, :3] * alpha + torch.full_like(argb[:, :3], fill_value=bg_color) * (
            1 - alpha
        )  # (B,3,H,W) RGB tensor

    def _save_and_backup(self, save_every, backup_every):
        os.makedirs(self.model_save_loc, exist_ok=True)
        os.makedirs(os.path.join(self.save_loc, "backups"), exist_ok=True)

        if(self.steps_done % save_every == 0 and self.steps_done > 0):
            self.model.save_model(os.path.join(self.model_save_loc, f"{self.run_name}.pt"))

        if self.steps_done % backup_every == 0 and self.steps_done > 0:
            self.model.save_model(
                os.path.join(self.save_loc, "backups", f"{self.run_name}_model_{self.steps_done / 1000:.1f}k.pt")
            )
        super()._save_and_backup(self.steps_done, save_every, backup_every)


    def _load_if_exists(self):
        to_load = os.path.join(self.save_loc, f"{self.run_name}.state")
        if(os.path.exists(to_load)):
            self.load_state(to_load)

        print(f'Successfully loaded existing model at {to_load}')

    @torch.no_grad()
    def log_progress(self, loss, batch, state):
        """
            Log the progress of the training.
            
            Args:
            loss : (B,) this step batched loss
            batch : (B,C,H,W) batch of input samples
            state : (B,C,H,W) batch of evolved samples
        """
        self.step_loss.append(loss.mean().item())

        if self.do_step_log:
            self.logger.log(
                {"loss/logl2": sum([math.log10(ell) for ell in self.step_loss]) / len(self.step_loss)},
                commit=False,
            )
            self.logger.log({"loss/l2": sum(self.step_loss) / len(self.step_loss)}, commit=False)
            self.logger.log({"metrics/lr": self.scheduler.get_last_lr()[0]}, commit=False)
            before_after = torch.cat((batch[0:8], state[0:8]), dim=0)  # (8,C,H,W)
            before_after = gridify(
                self.to_rgb(before_after, bg_color=0.6), max_width=400, columns=8
            )  # (3,H',W') grid of images

            before_after = wandb.Image(before_after.cpu(), caption=f"Up : Before, Down : After")

            self.logger.log({"evolved": before_after}, commit=False)
            self._update_x_axis()
            self.step_loss = []

def prepare_img(img_path: str, tarsize: tuple, pad: int = 4):
    # Prepare targets and seeds
    target = Image.open(img_path).convert("RGBA")  # (C,H,W)

    w, h = target.size

    transfo = transforms.Compose([transforms.ToTensor(), transforms.Resize(tarsize), transforms.Pad(pad, fill=0)])
    tensimg = transfo(target)
    tensimg[:3] = (
        tensimg[:3] * tensimg[3:4]
    )  # Premultiply by alpha, sometimes have non-zero RGB values where alpha is 0
    return tensimg
