import os
import wandb
import torch
from torch.optim import lr_scheduler
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, CenterCrop
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

import sys
sys.path.append("../")
from ndm import NDM, train_epoch
from celeba import CelebADataset as CelebA
from unet_openai.unet import UNetModel


class CelebaCustomDataset(CelebA):
    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image

def create_celeba_dataloader(config):
    transform = Compose([Resize(config["image_size"]),
                         CenterCrop(config["image_size"]),
                         ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        )
    dataset = CelebaCustomDataset(
        transform=transform,
        root_dir='/home/tem_shett/porn/CelebA/celeba/celeba',
    )
    loader = DataLoader(dataset, config["train_batch_size"], shuffle=True)
    return loader

def create_optimizer(ndm, config):
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
    return optimizer

def show_images(images, return_image, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().permute(0,2,3,1).numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(14, 14))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow((255*(images[idx]+1)/2).astype('uint8'))
                idx += 1
    fig.suptitle(title, fontsize=30)

    if return_image:
        # Wrap the Matplotlib figure in a wandb.Image
        wandb_img = wandb.Image(fig, caption=title)
        plt.close(fig)
        return wandb_img
    else:
        # Showing the figure
        plt.show()


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

    dataloader = create_celeba_dataloader(config)

    model = UNetModel(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    )

    print(f'UNetModel number of parameters: {sum([p.numel() for p in model.parameters()])}')

    model_F = UNetModel(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    )

    # Load pretrained model if specified
    next_epoch_num = 0
    if config["load_pretrained"]:
        assert not config.get("was_learnable", False) or config["schedule_config"].get("learnable",
                                                                                       False)
        need_switch = config.get("was_learnable", False) ^ config["schedule_config"].get(
            "learnable", False)
        config["schedule_config"]["learnable"] = config["was_learnable"]
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
        optimizer = create_optimizer(ndm, config)

        if config["pretrained_checkpoint_path"] is None:
            raise ValueError(
                "pretrained_checkpoint_path must be specified when load_pretrained is True"
            )
        if config["pretrained_run_id"]:
            print(f"Loading model from wandb run {config['pretrained_run_id']}")
            model_path = wandb.restore(
                config["pretrained_checkpoint_path"],
                run_path=f"{config['pretrained_run_id']}",
            )
            print(ndm.alphas_cumprod)
            print(model_path.name)
            ckpt = torch.load(model_path.name, map_location=device)
            ndm.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            next_epoch_num = ckpt['next_epoch']
        elif config["pretrained_checkpoint_path"]:
            print(f"Loading model from local path {config['pretrained_checkpoint_path']}")
            ckpt = torch.load(config["pretrained_checkpoint_path"], map_location=device)
            ndm.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            next_epoch_num = ckpt['next_epoch']
        else:
            raise ValueError(
                "Either pretrained_run_id or pretrained_checkpoint_path must be specified when load_pretrained is True"
            )
        if need_switch:
            ndm.alphas_cumprod = nn.Parameter(ndm.alphas_cumprod, requires_grad=True)
            config["schedule_config"]["learnable"] = True
        print('here', need_switch, ndm.alphas_cumprod)
    else:
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
        optimizer = create_optimizer(ndm, config)

    if config["wandb_logging"]:
        wandb.watch(ndm, log_freq=100)

    # Calculate total steps and create scheduler
    total_training_steps = len(dataloader) * config["num_epochs"]

    def lr_lambda(current_step):
        if current_step < config["warmup_steps"]:
            return float(current_step) / float(max(1, config["warmup_steps"]))

        decay_factor = max(
            0.0,
            float(total_training_steps - current_step) / float(
                max(1, total_training_steps - config["warmup_steps"]))
        )
        return decay_factor * (1.0 - config["final_lr"] / config["learning_rate"]) + config[
            "final_lr"] / config["learning_rate"]

    # Using LambdaLR for more explicit linear decay control
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0
    frames = []
    losses = []
    epoch_losses = []
    print("Training model...")
    for epoch in range(next_epoch_num, config["num_epochs"]):
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
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

        if epoch % config["save_images_step"] == 0 or epoch == config["num_epochs"] - 1:
            # generate data with the model to later visualize the learning process
            sample = ndm.sample(config["eval_batch_size"], (3, config["image_size"], config["image_size"]))
            # frames.append(sample.cpu().numpy())

            if do_plots:
                plt.figure(figsize=(16, 8))

                plt.subplot(1, 2, 1)
                plt.title("Loss per step")
                plt.plot(list(range(len(losses))), losses)
                plt.yscale("log")

                plt.subplot(1, 2, 2)
                plt.title("Loss per epoch")
                plt.plot(list(range(len(epoch_losses))), epoch_losses)
                plt.yscale("log")

                # Log the figure to wandb
                if config["wandb_logging"]:
                    wandb.log(
                        {"training_visualization": show_images(sample, return_image=True)},
                        step=epoch,
                    )
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()

                show_images(sample, return_image=False)

        if epoch % config["save_model_step"] == 0 or epoch == config["num_epochs"] - 1:
            print("Saving model...")
            outdir = f"exps/{config['experiment_name']}"
            os.makedirs(outdir, exist_ok=True)

            torch.save({
                'next_epoch': epoch + 1,
                'model_state_dict': ndm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{outdir}/checkpoint_epoch_{epoch}.pth")

    # save model to wandb
    if config["wandb_logging"]:
        wandb.save(f"{outdir}/checkpoint_epoch_{config['num_epochs'] - 1}.pth")
    #
    # print("Saving images...")
    # imgdir = f"{outdir}/images"
    # os.makedirs(imgdir, exist_ok=True)
    # frames = np.stack(frames)
    # xmin, xmax = -6, 6
    # ymin, ymax = -6, 6
    # for i, frame in enumerate(frames):
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(frame[:, 0], frame[:, 1])
    #     plt.xlim(xmin, xmax)
    #     plt.ylim(ymin, ymax)
    #     plt.savefig(f"{imgdir}/{i:04}.png")
    #     plt.close()
    # print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))
    #
    # print("Saving frames...")
    # np.save(f"{outdir}/frames.npy", frames)
    #
    # Finish wandb run
    if config["wandb_logging"]:
        wandb.finish()


if __name__ == "__main__":
    config = {
        "experiment_name": "ndm_images_1000steps",
        "wandb_logging": False,
        "image_size": 16,
        "train_batch_size": 64,
        "eval_batch_size": 16,
        "num_epochs": 4,
        "learning_rate": 3e-4,
        "warmup_steps": 0,
        "num_timesteps": 1000,
        "schedule_config": {"type": "cosine", "min_alpha": 0.0001, "max_alpha": 0.9999,
                            "learnable": False},
        # "schedule_config": {"type": "linear", "beta_start": 0.0001, "beta_end": 0.02, "learnable": False},
        "save_images_step": 1,
        "save_model_step": 1,
        "gradient_clipping": None,
        "importance_sampling_batch_size": None,
        "uniform_prob": 0.001,
        "optimizer_type": "adamw",
        "momentum": 0.9,
        "weight_decay": 0.00001,
        # New parameters for model loading
        "load_pretrained": True,
        # "pretrained_run_id": "ndm/esjakfmk",  # wandb run ID to load model from
        "pretrained_run_id": None,
        # "pretrained_model_path": "exps/ndm_1000steps/model.pth",  # local path to load model from (alternative to wandb)
        "pretrained_checkpoint_path": "exps/ndm_images_1000steps/checkpoint_epoch_2.pth",
        "was_learnable": False,
        "predict_noise": True,
        "ddim_sampling": False,
        "final_lr": 1e-10
    }

    run(config, do_plots=True)
