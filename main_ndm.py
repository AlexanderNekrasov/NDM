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
    if config["wandb_logging"]:
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
        predict_noise=config["predict_noise"],
        ddim_sampling=config["ddim_sampling"]
    ).to(device)

    # Load pretrained model if specified
    if config["load_pretrained"]:
        if config["pretrained_model_path"] is None:
            raise ValueError(
                "pretrained_model_path must be specified when load_pretrained is True"
            )
        if config["pretrained_run_id"]:
            print(f"Loading model from wandb run {config['pretrained_run_id']}")
            model_path = wandb.restore(
                config["pretrained_model_path"],
                run_path=f"{config['pretrained_run_id']}",
            )
            ndm.load_state_dict(torch.load(model_path.name, map_location=device))
        elif config["pretrained_model_path"]:
            print(f"Loading model from local path {config['pretrained_model_path']}")
            ndm.load_state_dict(
                torch.load(config["pretrained_model_path"], map_location=device)
            )
        else:
            raise ValueError(
                "Either pretrained_run_id or pretrained_model_path must be specified when load_pretrained is True"
            )

    if config["wandb_logging"]:
        wandb.watch(ndm, log_freq=100)

    # Initialize optimizer based on config
    optimizer_parameters = list(ndm.model.parameters()) + list(ndm.model_F.parameters())
    if config["schedule_config"].get("learnable", False):
        optimizer_parameters += [ndm.alphas_cumprod]
    if config["optimizer_type"].lower() == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_parameters,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer_type"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer_type']}")

    # Calculate total steps and create scheduler
    total_training_steps = len(dataloader) * config["num_epochs"]
    def lr_lambda(current_step):
        if current_step < config["warmup_steps"]:
            return float(current_step) / float(max(1, config["warmup_steps"]))
        
        decay_factor = max(
            0.0,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - config["warmup_steps"]))
        )
        return decay_factor * (1.0 - config["final_lr"] / config["learning_rate"]) + config["final_lr"] / config["learning_rate"]
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
        if config["schedule_config"].get("learnable", False):
            print("alphas_cumprod after train_epoch:", ndm.alphas_cumprod)
        losses.extend(cur_losses)
        epoch_loss = np.mean(cur_losses)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch} finished with loss: {epoch_loss}")

        # Log metrics to wandb
        if config["wandb_logging"]:
            wandb.log(
                {
                    "epoch": epoch + config["pretrained_epochs"],
                    "loss": epoch_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch + config["pretrained_epochs"],
            )

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
                if config["wandb_logging"]:
                    wandb.log(
                        {"training_visualization": wandb.Image(plt)},
                        step=epoch + config["pretrained_epochs"],
                    )
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

    print("Saving model...")
    outdir = f"exps/{config['experiment_name']}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(ndm.state_dict(), f"{outdir}/model.pth")
    # save model to wandb
    if config["wandb_logging"]:
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
    if config["wandb_logging"]:
        wandb.finish()


if __name__ == "__main__":
    # Configuration
    config = {
        "experiment_name": "ndm_1000steps",
        "wandb_logging": False,
        "dataset": "checkerboard",
        "train_batch_size": 256,
        "eval_batch_size": 1000,
        "num_epochs": 1000,
        "learning_rate": 1e-4,
        "warmup_steps": 1000,
        "num_timesteps": 30,
        "schedule_config": {"type": "cosine", "min_alpha": 0.0001, "max_alpha": 0.9999, "learnable": False},
        # "schedule_config": {"type": "linear", "beta_start": 0.0001, "beta_end": 0.02, "learnable": False},
        "embedding_size": 128,
        "hidden_size": 512,
        "hidden_layers": 5,
        "save_images_step": 1,
        "gradient_clipping": None,
        "dataset_size": 80000,
        "importance_sampling_batch_size": None,
        "uniform_prob": 0.001,
        "optimizer_type": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        # New parameters for model loading
        "load_pretrained": False,
        # "pretrained_run_id": "ndm/esjakfmk",  # wandb run ID to load model from
        "pretrained_run_id": None,
        # "pretrained_model_path": "exps/ndm_1000steps/model.pth",  # local path to load model from (alternative to wandb)
        "pretrained_model_path": None,
        "pretrained_epochs": 0,  # number of epochs the pretrained model was trained for
        "predict_noise": True,
        "ddim_sampling": False,
        "final_lr": 1e-10
    }

    run(config, do_plots=True)
