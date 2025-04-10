import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
import numpy as np


class DDPM(nn.Module):
    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):

        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.register_buffer(
            "betas",
            torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(torch.concat([torch.ones(1), self.alphas]), axis=0),
        )

    def reconstruct_x0(self, x_t, t, noise):
        s1 = torch.sqrt(1 / self.alphas_cumprod[t])
        s2 = torch.sqrt(1 / self.alphas_cumprod[t] - 1)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = (
            self.betas[t - 1]
            * torch.sqrt(self.alphas_cumprod[t - 1])
            / (1.0 - self.alphas_cumprod[t])
        )
        s2 = (
            (1.0 - self.alphas_cumprod[t - 1])
            * torch.sqrt(self.alphas[t - 1])
            / (1.0 - self.alphas_cumprod[t])
        )
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 1:
            return 0

        variance = (
            self.betas[t - 1]
            * (1.0 - self.alphas_cumprod[t - 1])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 1:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = torch.sqrt(self.alphas_cumprod[timesteps])
        s2 = torch.sqrt(1 - self.alphas_cumprod[timesteps])

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def sample(self, n_samples):
        self.model.eval()
        device = next(self.parameters()).device
        sample = torch.randn(n_samples, 2, device=device)
        timesteps = list(range(1, len(self) + 1))[::-1]
        for t in tqdm(timesteps):
            t = torch.from_numpy(np.repeat(t, n_samples)).long().to(device)
            with torch.no_grad():
                residual = self.model(sample, t)
            sample = self.step(residual, t[0], sample)
        return sample

    def __len__(self):
        return self.num_timesteps


def train_epoch(ddpm, dataloader, optimizer, scheduler, global_step):
    ddpm.model.train()
    epoch_losses = []
    device = next(ddpm.parameters()).device

    for step, batch in enumerate(dataloader):
        batch = batch[0].to(device)
        noise = torch.randn(batch.shape, device=device)
        timesteps = (
            torch.randint(1, ddpm.num_timesteps + 1, (batch.shape[0],))
            .long()
            .to(device)
        )

        noisy = ddpm.add_noise(batch, noise, timesteps)
        noise_pred = ddpm.model(noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.detach().item())
        global_step += 1
    return global_step, epoch_losses
