import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
import numpy as np
from jaxtyping import Int, Float
from torch import Tensor


class DDPM(nn.Module):
    def __init__(
        self,
        model,
        schedule_config,
        num_timesteps=1000,
        importance_sampling_batch_size=None,
        uniform_prob=0.001,
    ):

        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.importance_sampling_batch_size = importance_sampling_batch_size

        # Initialize importance sampling buffers if needed
        if importance_sampling_batch_size is not None:
            self.L_t = torch.zeros(num_timesteps, importance_sampling_batch_size)
            self.L_t_counts = torch.zeros(num_timesteps, dtype=torch.long)
            self.L_t_ptr = torch.zeros(num_timesteps, dtype=torch.long)
            self.uniform_prob = uniform_prob
        if schedule_config["type"] == "linear":
            # example schedule_config: {"type": "linear", "beta_start": 0.0001, "beta_end": 0.02}
            betas: Float[Tensor, "{num_timesteps + 1}"] = torch.linspace(
                schedule_config["beta_start"],
                schedule_config["beta_end"],
                num_timesteps + 1,
                dtype=torch.float32,
            )
            alphas: Float[Tensor, "{num_timesteps + 1}"] = 1.0 - betas
            alphas_cumprod: Float[Tensor, "{num_timesteps + 1}"] = torch.cumprod(
                torch.cat([alphas]), axis=0
            )
        elif schedule_config["type"] == "cosine":
            # example schedule_config: {"type": "cosine", "min_alpha": 0.0001, "max_alpha": 0.9999}
            ts: Int[Tensor, "{num_timesteps + 1}"] = torch.linspace(
                0, 1, num_timesteps + 1
            )
            min_alpha = schedule_config["min_alpha"]
            max_alpha = schedule_config["max_alpha"]
            alphas_cumprod: Float[Tensor, "{num_timesteps + 1}"] = (
                torch.cos(ts * (torch.pi / 2)) ** 2 * (max_alpha - min_alpha)
                + min_alpha
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_config['type']}")

        print("alphas_cumpod:", alphas_cumprod)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def reconstruct_x0(
        self,
        x_t: Float[Tensor, "B 2"],
        t: Int[Tensor, "B"],
        noise: Float[Tensor, "B 2"],
    ):
        s1 = torch.sqrt(1 / self.alphas_cumprod[t])
        s2 = torch.sqrt(1 / self.alphas_cumprod[t] - 1)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(
        self,
        x_0: Float[Tensor, "B 2"],
        x_t: Float[Tensor, "B 2"],
        t: Int[Tensor, "B"] | Int[Tensor, ""],
    ):
        B = x_0.shape[0]
        assert torch.is_tensor(t)
        assert t.shape == (B,) or t.shape == ()
        assert (1 <= t).all() and (t <= self.num_timesteps).all()

        a_s = self.alphas_cumprod[t - 1]
        a_t = self.alphas_cumprod[t]
        sigma_s = torch.sqrt(1 - a_s)
        sigma_t = torch.sqrt(1 - a_t)
        sigma_st = torch.sqrt(
            (sigma_t**2 - a_t / a_s * sigma_s**2) * sigma_s**2 / sigma_t**2
        )

        s0 = torch.sqrt(self.alphas_cumprod[t - 1])
        s1 = -torch.sqrt(sigma_s**2 - sigma_st**2) / sigma_t * torch.sqrt(a_t)
        s2 = torch.sqrt(sigma_s**2 - sigma_st**2) / sigma_t

        s0 = s0.reshape(-1, 1)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        mu = (s0 + s1) * x_0 + s2 * x_t
        return mu

    def get_variance(self, t: Int[Tensor, "B"]):
        a_s = self.alphas_cumprod[t - 1]
        a_t = self.alphas_cumprod[t]
        sigma_s = torch.sqrt(1 - a_s)
        sigma_t = torch.sqrt(1 - a_t)
        sigma_st = torch.sqrt(
            (sigma_t**2 - a_t / a_s * sigma_s**2) * sigma_s**2 / sigma_t**2
        )
        variance = sigma_st**2
        return variance

    def step(
        self,
        model_output: Float[Tensor, "B 2"],
        timestep: Int[Tensor, "B"],
        sample: Float[Tensor, "B 2"],
    ):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 1:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
        self,
        x_start: Float[Tensor, "B 2"],
        x_noise: Float[Tensor, "B 2"],
        timesteps: Int[Tensor, "B"],
    ):
        s1 = torch.sqrt(self.alphas_cumprod[timesteps])
        s2 = torch.sqrt(1 - self.alphas_cumprod[timesteps])

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def sample(self, n_samples: int):
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


def train_epoch(
    ddpm: DDPM,
    dataloader,
    optimizer,
    scheduler,
    global_step,
    gradient_clipping=0.1,
    use_simplified_loss=False,
):
    importance_sampling = ddpm.importance_sampling_batch_size is not None

    ddpm.model.train()
    epoch_losses = []
    device = next(ddpm.parameters()).device

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch[0].to(device)
        noise = torch.randn(batch.shape, device=device)

        if (
            importance_sampling
            and ddpm.L_t_counts.min() >= ddpm.importance_sampling_batch_size
        ):
            weights = torch.sqrt(torch.mean(ddpm.L_t**2, dim=1))
            weights = weights / weights.sum()

            # Add numerical stability checks
            epsilon = 1e-6
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print(
                    "Warning: Weights contain NaN or Inf values, using uniform weights"
                )
                weights = (
                    torch.ones(ddpm.num_timesteps, device=device) / ddpm.num_timesteps
                )
            elif torch.abs(weights.sum() - 1.0) > epsilon:
                print("Warning: Weights sum is not close to 1, using uniform weights")
                weights = (
                    torch.ones(ddpm.num_timesteps, device=device) / ddpm.num_timesteps
                )

            weights = weights * (1 - ddpm.uniform_prob) + ddpm.uniform_prob / len(
                weights
            )
        else:
            weights = torch.ones(ddpm.num_timesteps, device=device)
            weights = weights / weights.sum()
        weights = weights.to(device)
        timesteps = (
            torch.multinomial(weights, num_samples=batch.shape[0], replacement=True)
            + 1
        )

        a_s = ddpm.alphas_cumprod[timesteps - 1]
        a_t = ddpm.alphas_cumprod[timesteps]
        sigma_s = torch.sqrt(1 - a_s)
        sigma_t = torch.sqrt(1 - a_t)
        sigmast_squared = (
            (sigma_t**2 - a_t / a_s * sigma_s**2) * sigma_s**2 / sigma_t**2
        )

        noisy = ddpm.add_noise(batch, noise, timesteps)
        noise_pred = ddpm.model(noisy, timesteps)
        loss_simple = F.mse_loss(noise_pred, noise)
        mu = ddpm.q_posterior(batch, noisy, timesteps)
        xhat = ddpm.reconstruct_x0(noisy, timesteps, noise_pred)
        mu_hat = ddpm.q_posterior(xhat, noisy, timesteps)
        loss_nll = mu - mu_hat
        loss_nll = loss_nll**2
        loss_nll = loss_nll.view(loss_nll.shape[0], -1).sum(dim=1)
        loss_nll = loss_nll / (2 * sigmast_squared)
        loss_mean = (loss_nll / weights[timesteps - 1]).mean()

        optimizer.zero_grad()
        if use_simplified_loss:
            loss_simple.backward()
        else:
            loss_mean.backward()
        if gradient_clipping is not None:
            nn.utils.clip_grad_norm_(ddpm.model.parameters(), gradient_clipping)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss_mean.detach().item())
        global_step += 1

        if importance_sampling:
            loss_nll = loss_nll.detach().cpu()
            timesteps = timesteps.cpu()
            for t, l in zip(timesteps, loss_nll):
                idx = t - 1
                ptr = ddpm.L_t_ptr[idx]
                ddpm.L_t[idx, ptr] = l
                ddpm.L_t_ptr[idx] = (ptr + 1) % ddpm.importance_sampling_batch_size
                ddpm.L_t_counts[idx] = min(
                    ddpm.L_t_counts[idx] + 1, ddpm.importance_sampling_batch_size
                )

    return global_step, epoch_losses
