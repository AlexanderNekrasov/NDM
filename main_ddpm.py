import argparse
import os

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler

import datasets
from ddpm import DDPM, train_epoch
from network import MLP


# Configuration
config = {
    "experiment_name": "base_1000steps",
    "dataset": "checkerboard",
    "train_batch_size": 256,
    "eval_batch_size": 1000,
    "num_epochs": 300,
    "learning_rate": 1e-4,
    "num_timesteps": 1000,
    "embedding_size": 128,
    "hidden_size": 512,
    "hidden_layers": 5,
    "save_images_step": 20,
}

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

dataset = datasets.get_dataset(config["dataset"])
dataloader = DataLoader(
    dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=True
)

model = MLP(
    hidden_size=config["hidden_size"],
    hidden_layers=config["hidden_layers"],
    emb_size=config["embedding_size"],
)


ddpm = DDPM(model=model, num_timesteps=config["num_timesteps"]).to(device)

optimizer = torch.optim.AdamW(
    ddpm.model.parameters(),
    lr=config["learning_rate"],
)

# Calculate total steps and create scheduler
total_training_steps = len(dataloader) * config["num_epochs"]
final_lr = 1e-8
lr_lambda = (
    lambda current_step: max(
        0.0,
        float(total_training_steps - current_step)
        / float(max(1, total_training_steps)),
    )
    * (1.0 - final_lr / config["learning_rate"])
    + final_lr / config["learning_rate"]
)
# Using LambdaLR for more explicit linear decay control
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

global_step = 0
frames = []
losses = []
print("Training model...")
for epoch in range(config["num_epochs"]):
    print(f"Epoch {epoch}")
    global_step, epoch_losses = train_epoch(
        ddpm, dataloader, optimizer, scheduler, global_step
    )
    losses.extend(epoch_losses)
    print(f"Epoch {epoch} finished with loss: {np.mean(epoch_losses)}")
    # plot losses in log scale

    if epoch % config["save_images_step"] == 0 or epoch == config["num_epochs"] - 1:
        # generate data with the model to later visualize the learning process
        sample = ddpm.sample(config["eval_batch_size"])
        frames.append(sample.cpu().numpy())
        # plt.figure(figsize=(20, 10))

        # plt.subplot(1, 2, 1)
        # plt.plot(list(range(len(losses))), losses)
        # plt.yscale("log")

        # plt.subplot(1, 2, 2)
        # plt.xlim(-6, 6)
        # plt.ylim(-6, 6)
        # plt.scatter(frames[-1][:, 0], frames[-1][:, 1])

        # plt.show()

print("Saving model...")
outdir = f"exps/{config['experiment_name']}"
os.makedirs(outdir, exist_ok=True)
torch.save(ddpm.model.state_dict(), f"{outdir}/model.pth")

print("Saving images...")
imgdir = f"{outdir}/images"
os.makedirs(imgdir, exist_ok=True)
frames = np.stack(frames)
xmin, xmax = -6, 6
ymin, ymax = -6, 6
for i, frame in enumerate(frames):
    plt.figure(figsize=(10, 10))
    plt.scatter(frame[:, 0], frame[:, 1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig(f"{imgdir}/{i:04}.png")
    plt.close()
print("Saving loss as numpy array...")
np.save(f"{outdir}/loss.npy", np.array(losses))

print("Saving frames...")
np.save(f"{outdir}/frames.npy", frames)
