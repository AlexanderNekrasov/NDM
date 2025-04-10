import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from ndm import NDM
from network import MLP
from datasets import get_dataset

# Configuration matching the training run
config = {
    "num_timesteps": 1000,
    "embedding_size": 128,
    "hidden_size": 512,
    "hidden_layers": 5,
    "experiment_name": "ndm_1000steps_clippingNone",
    "n_samples": 100,
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

    model_F = MLP(
        hidden_size=config["hidden_size"],
        hidden_layers=config["hidden_layers"],
        emb_size=config["embedding_size"],
    )

    ndm = NDM(model=model, model_F=model_F, num_timesteps=config["num_timesteps"]).to(
        device
    )

    # Load model weights
    model_path = f"exps/{config['experiment_name']}/model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    # The saved checkpoint contains the state_dict of the inner model (MLP)
    ndm.model.load_state_dict(torch.load(model_path, map_location=device))
    ndm.eval()  # Set the DDPM wrapper to evaluation mode

    # get points from dataset
    dataset = get_dataset(config["dataset"])
    points = [dataset[i][0] for i in range(len(dataset))]
    points = torch.stack(points, dim=0).to(device)
    with torch.no_grad():
        transformed_samples = (
            ndm.model_F(
                points,
                torch.ones((points.shape[0],), device=device)
                * config["num_timesteps"]
                * 0,
            )
            .cpu()
            .detach()
            .numpy()
        )

    # Sample points
    # print(f"Sampling {config['n_samples']} points...")
    # samples = ndm.sample(config["n_samples"])
    # samples_np = samples.cpu().numpy()

    # Save samples
    # output_path = f"exps/{config['experiment_name']}/samples.npy"
    # print(f"Saving samples to {output_path}")
    # np.save(output_path, samples_np)

    # Plot samples
    print("Plotting samples...")
    plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    # # plt.scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.5)
    # plt.title(f"Samples from {config['experiment_name']}")
    # plt.xlabel("X1")
    # plt.ylabel("X2")

    # plt.subplot(2, 1, 2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(transformed_samples[:, 0], transformed_samples[:, 1], s=1, alpha=0.5)
    plt.title(f"Samples from {config['experiment_name']}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate_ddpm()
