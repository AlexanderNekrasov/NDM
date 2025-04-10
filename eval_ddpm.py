import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from ddpm import DDPM
from network import MLP

# Configuration matching the training run
config = {
    "num_timesteps": 1000,
    "embedding_size": 128,
    "hidden_size": 512,
    "hidden_layers": 5,
    "experiment_name": "base_1000steps",
    "n_samples": 10000,
    "dataset": "checkerboard",
}


def evaluate_ddpm():
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Instantiate models
    model = MLP(
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        emb_size=config["embedding_size"],
    )

    ddpm = DDPM(model=model, num_timesteps=config["num_timesteps"]).to(device)

    # Load model weights
    model_path = f"exps/{config['experiment_name']}/model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    # The saved checkpoint contains the state_dict of the inner model (MLP)
    ddpm.model.load_state_dict(torch.load(model_path, map_location=device))
    ddpm.eval()  # Set the DDPM wrapper to evaluation mode

    # Sample points
    print(f"Sampling {config['n_samples']} points...")
    samples = ddpm.sample(config["n_samples"])
    samples_np = samples.cpu().numpy()

    # Plot samples
    print("Plotting samples...")
    plt.figure(figsize=(8, 8))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.5)
    plt.title(f"Samples from {config['experiment_name']}")
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.show()

    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate_ddpm()
