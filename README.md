# Neural Diffusion Models (NDM)

This repository contains an implementation of the [Neural Diffusion Models](https://arxiv.org/abs/2310.08337) paper. The implementation includes both the original DDPM (Denoising Diffusion Probabilistic Models) and the novel NDM approach.

## Project Structure

```
.
├── main_ndm.py          # Main training script for NDM
├── main_ddpm.py         # Main training script for DDPM
├── ndm.py              # NDM model implementation
├── ddpm.py             # DDPM model implementation
├── network.py          # Neural network architectures
├── datasets.py         # Dataset loading and preprocessing
├── eval_ndm.py         # Evaluation script for NDM
├── eval_ddpm.py        # Evaluation script for DDPM
├── FID.py              # Fréchet Inception Distance calculation
├── visualize_transform.py # Visualization utilities
├── visualizations/     # Directory for generated visualizations
└── image_generation/   # Directory for image generation scripts
    ├── main_ndm_images.py  # Main script for generating images with NDM
    ├── ndm.py             # NDM implementation for image generation
    ├── ndm_images.ipynb   # Jupyter notebook for image generation
    ├── celeba.py          # CelebA dataset handling
    └── unet_openai/       # OpenAI's UNet implementation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- wandb (for experiment tracking)

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the NDM model:
```bash
python main_ndm.py
```

To train the DDPM model:
```bash
python main_ddpm.py
```

### Evaluation

To evaluate the NDM model:
```bash
python eval_ndm.py
```

To evaluate the DDPM model:
```bash
python eval_ddpm.py
```

### Visualization

To generate visualizations of the diffusion process:
```bash
python visualize_transform.py
```


