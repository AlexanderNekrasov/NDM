from ndm import NDM
import wandb
import torch
import matplotlib.pyplot as plt
from network import MLP
from datasets import get_dataset
from torch.utils.data import DataLoader
import os

def visualize_transform(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
    model_path = wandb.restore(
        config["pretrained_model_path"],
        run_path=f"{config['pretrained_run_id']}",
    )
    ndm.load_state_dict(torch.load(model_path.name, map_location=device))
    ndm.eval()

    dataset = get_dataset(config["dataset"], config["dataset_size"])
    dataloader = DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False)
    
    # Get a batch of data
    batch = next(iter(dataloader))
    if isinstance(batch, list):
        batch = batch[0]
    batch = batch.to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Create figure with subplots for F-transformed data
    n_timesteps = min(6, ndm.num_timesteps + 1)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(6*n_timesteps, 6))
    if n_timesteps == 1:
        axes = [axes]
    
    # Transform and plot at different timesteps
    timesteps = [0] + [i * (ndm.num_timesteps // (n_timesteps-1)) for i in range(1, n_timesteps)]
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            transformed = ndm.F(batch, torch.tensor(t, device=device).repeat(batch.shape[0]))
            transformed = transformed.cpu().numpy()
            
            ax = axes[i]
            ax.scatter(transformed[:, 0], transformed[:, 1], s=1, alpha=0.5)
            ax.set_title(f"t = {t}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
    
    plt.tight_layout()
    plt.savefig("visualizations/f_transformed.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create figure with subplots for noise-added data
    fig, axes = plt.subplots(1, n_timesteps, figsize=(6*n_timesteps, 6))
    if n_timesteps == 1:
        axes = [axes]
    
    # Transform and plot at different timesteps
    timesteps = [0] + [i * (ndm.num_timesteps // (n_timesteps-1)) for i in range(1, n_timesteps)]
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            transformed = ndm.add_noise(batch, torch.randn_like(batch), torch.tensor(t, device=device).repeat(batch.shape[0]))
            transformed = transformed.cpu().numpy()
            
            ax = axes[i]
            ax.scatter(transformed[:, 0], transformed[:, 1], s=1, alpha=0.5)
            ax.set_title(f"t = {t}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
    
    plt.tight_layout()
    plt.savefig("visualizations/noise_added.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    config = {
        "dataset": "checkerboard",
        "eval_batch_size": 10000,
        "num_timesteps": 10,
        "embedding_size": 256*3,
        "schedule_config": {"type": "cosine", "min_alpha": 0.0001, "max_alpha": 0.9999, "learnable": True},
        "hidden_size": 256*4,
        "hidden_layers": 7,
        "dataset_size": 80000,
        "importance_sampling_batch_size": None,
        "uniform_prob": 0.001,
        "predict_noise": True,
        "ddim_sampling": False,
        "pretrained_run_id": "ndm/l6w3l587",  # wandb run ID to load model from
        "pretrained_model_path": "exps/ndm_1000steps/model.pth",  # local path to load model from (alternative to wandb)
    }
    wandb.login()
    visualize_transform(config)