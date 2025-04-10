import argparse
import os

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
import wandb

import datasets
from ndm import NDM, train_epoch
from network import MLP


def run(config, do_plots=False):
    # Initialize wandb
    wandb.init(
        entity="alexxela12345-hse-university",
        project="ndm",
        # name=config["experiment_name"],
        config=config,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = datasets.get_dataset(config["dataset"], n=config["dataset_size"])
    dataloader = DataLoader(
        dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=True
    )

    model = MLP(
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        emb_size=config["embedding_size"],
    )

    model_F = MLP(
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        emb_size=config["embedding_size"],
    )

    ndm = NDM(
        model=model,
        model_F=model_F,
        schedule_config=config["schedule_config"],
        num_timesteps=config["num_timesteps"],
        importance_sampling_batch_size=config["importance_sampling_batch_size"],
        uniform_prob=config["uniform_prob"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(ndm.model.parameters()) + list(ndm.model_F.parameters()),
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
    epoch_losses = []
    print("Training model...")
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch}")
        global_step, cur_losses = train_epoch(
            ndm,
            dataloader,
            optimizer,
            scheduler,
            global_step,
            config["gradient_clipping"],
        )
        losses.extend(cur_losses)
        epoch_loss = np.mean(cur_losses)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch} finished with loss: {epoch_loss}")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }, step=epoch)

        if epoch % config["save_images_step"] == 0 or epoch == config["num_epochs"] - 1:
            # generate data with the model to later visualize the learning process
            sample = ndm.sample(config["eval_batch_size"])
            frames.append(sample.cpu().numpy())
            
            if do_plots:
                plt.figure(figsize=(40, 10))

                plt.subplot(1, 5, 1)
                plt.title("Loss per step")
                plt.plot(list(range(len(losses))), losses)
                plt.yscale("log")

                plt.subplot(1, 5, 2)
                plt.title("Loss per epoch")
                plt.plot(list(range(len(epoch_losses))), epoch_losses)
                plt.yscale("log")

                plt.subplot(1, 5, 3)
                plt.title("Dataset samples")
                points = [dataset[i][0] for i in range(len(dataset))]
                points = torch.stack(points, dim=0)
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)

                plt.subplot(1, 5, 4)
                plt.title("Generated samples (cropped)")
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.scatter(frames[-1][:, 0], frames[-1][:, 1], s=7, alpha=1)

                plt.subplot(1, 5, 5)
                plt.title("Generated samples")
                plt.scatter(frames[-1][:, 0], frames[-1][:, 1], s=7, alpha=1)

                # Log the figure to wandb
                wandb.log({"training_visualization": wandb.Image(plt)}, step=epoch)
                plt.show()
                plt.close()

    print("Saving model...")
    outdir = f"exps/{config['experiment_name']}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(ndm.state_dict(), f"{outdir}/model.pth")
    # save model to wandb
    wandb.save(f"{outdir}/model.pth")

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
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    # Configuration
    config = {
        "experiment_name": "ndm_1000steps_clippingNone",
        "dataset": "checkerboard",
        "train_batch_size": 256,
        "eval_batch_size": 1000,
        "num_epochs": 300,
        "learning_rate": 1e-4,
        "num_timesteps": 1000,
        # "schedule_config": {"type": "cosine", "min_alpha": 0.0001, "max_alpha": 0.9999},
        "schedule_config": {"type": "linear", "beta_start": 0.0001, "beta_end": 0.02},
        "embedding_size": 128,
        "hidden_size": 512,
        "hidden_layers": 5,
        "save_images_step": 1,
        "gradient_clipping": None,
        "dataset_size": 80000,
        "importance_sampling_batch_size": 10,
        "uniform_prob": 0.001,
    }

    run(config, do_plots=False)
