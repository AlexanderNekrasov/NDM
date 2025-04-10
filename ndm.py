import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
import numpy as np


class NDM(nn.Module):
    def __init__(
        self,
        model,
        model_F,
        schedule_config,
        num_timesteps=1000,
        importance_sampling_batch_size=None,
        uniform_prob=0.001,
    ):

        super().__init__()
        self.model = model
        self.model_F = model_F
        self.num_timesteps = num_timesteps
        self.importance_sampling_batch_size = importance_sampling_batch_size

        # Initialize importance sampling buffers if needed
        if importance_sampling_batch_size is not None:
            self.register_buffer(
                "L_t", torch.zeros(num_timesteps, importance_sampling_batch_size)
            )
            self.register_buffer(
                "L_t_counts", torch.zeros(num_timesteps, dtype=torch.long)
            )
            self.register_buffer(
                "L_t_ptr", torch.zeros(num_timesteps, dtype=torch.long)
            )
            self.uniform_prob = uniform_prob
        if schedule_config["type"] == "linear":
            # example schedule_config: {"type": "linear", "beta_start": 0.0001, "beta_end": 0.02}
            betas = torch.linspace(
                schedule_config["beta_start"],
                schedule_config["beta_end"],
                num_timesteps + 1,
                dtype=torch.float32,
            )
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(torch.cat([alphas]), axis=0)
        elif schedule_config["type"] == "cosine":
            # example schedule_config: {"type": "cosine", "min_alpha": 0.0001, "max_alpha": 0.9999}
            ts = torch.linspace(0, 1, num_timesteps + 1)
            min_alpha = schedule_config["min_alpha"]
            max_alpha = schedule_config["max_alpha"]
            alphas_cumprod = (
                torch.cos(ts * (torch.pi / 2)) ** 2 * (max_alpha - min_alpha) + min_alpha
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_config['type']}")

        print("alphas_cumpod:", alphas_cumprod)
        # self.register_buffer("betas", betas)
        # self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def F(self, x, t):
        if torch.is_tensor(t):
            assert (0 <= t).all() and (t <= self.num_timesteps).all()
        else:
            assert 0 <= t and t <= self.num_timesteps
        t_scaled = t.view(-1, 1) / self.num_timesteps
        return (1 - t_scaled) * x + t_scaled * self.model_F(x, t)

    def reconstruct_x0(self, z_t, t, noise):
        s1 = torch.sqrt(1 / self.alphas_cumprod[t])
        s2 = torch.sqrt(1 / self.alphas_cumprod[t] - 1)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * z_t - s2 * noise

    def q_posterior(self, x_0, z_t, t):
        assert torch.is_tensor(t)
        assert t.shape == ()
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

        mu = (
            s0 * self.F(x_0, (t - 1).repeat(x_0.shape[0]))
            + s1 * self.F(x_0, t.repeat(x_0.shape[0]))
            + s2 * z_t
        )
        return mu

    def get_variance(self, t):
        a_s = self.alphas_cumprod[t - 1]
        a_t = self.alphas_cumprod[t]
        sigma_s = torch.sqrt(1 - a_s)
        sigma_t = torch.sqrt(1 - a_t)
        sigma_st = torch.sqrt(
            (sigma_t**2 - a_t / a_s * sigma_s**2) * sigma_s**2 / sigma_t**2
        )
        variance = sigma_st**2
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

        return s1 * self.F(x_start, timesteps) + s2 * x_noise

    @torch.no_grad()
    def sample(self, n_samples):
        self.model.eval()
        self.model_F.eval()
        device = next(self.parameters()).device
        sample = torch.randn(n_samples, 2, device=device)
        timesteps = list(range(1, self.num_timesteps + 1))[::-1]
        for t in tqdm(timesteps):
            t = torch.from_numpy(np.repeat(t, n_samples)).long().to(device)
            residual = self.model(sample, t)
            sample = self.step(residual, t[0], sample)
        return sample

    def __len__(self):
        return self.num_timesteps


def train_epoch(
    ndm: NDM, dataloader, optimizer, scheduler, global_step, gradient_clipping=0.1
):
    importance_sampling = ndm.importance_sampling_batch_size is not None

    ndm.model.train()
    ndm.model_F.train()
    epoch_losses = []
    device = next(ndm.parameters()).device

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch[0].to(device)
        noise = torch.randn(batch.shape, device=device)

        if importance_sampling and ndm.L_t_counts.min() >= ndm.importance_sampling_batch_size:
            weights = torch.sqrt(torch.mean(ndm.L_t**2, dim=1))
            weights = weights / weights.sum()
            weights = weights * (1 - ndm.uniform_prob) + ndm.uniform_prob / len(
                weights
            )
        else:
            weights = torch.ones(ndm.num_timesteps, device=device)
            weights = weights / weights.sum()
        timesteps = (
            torch.multinomial(
                weights, num_samples=batch.shape[0], replacement=True
            ).to(device)
            + 1
        )

        a_s = ndm.alphas_cumprod[timesteps - 1]
        a_t = ndm.alphas_cumprod[timesteps]
        sigma_s = torch.sqrt(1 - a_s)
        sigma_t = torch.sqrt(1 - a_t)
        sigmast_squared = (
            (sigma_t**2 - a_t / a_s * sigma_s**2) * sigma_s**2 / sigma_t**2
        )

        noisy = ndm.add_noise(batch, noise, timesteps)
        noise_pred = ndm.model(noisy, timesteps)
        xhat = ndm.reconstruct_x0(noisy, timesteps, noise_pred)

        coef_1 = torch.sqrt(ndm.alphas_cumprod[timesteps]).view(-1, 1)
        loss_1 = coef_1 * (ndm.F(batch, timesteps - 1) - ndm.F(xhat, timesteps - 1))

        coef_2 = (
            torch.sqrt(sigma_s**2 - sigmast_squared)
            / sigma_t
            * torch.sqrt(ndm.alphas_cumprod[timesteps])
        ).view(-1, 1)
        loss_2 = coef_2 * (ndm.F(xhat, timesteps) - ndm.F(batch, timesteps))

        loss = loss_1 + loss_2
        loss = loss**2
        loss = loss.view(loss.shape[0], -1).sum(dim=1)
        loss = 1 / (2 * sigmast_squared) * loss
        loss_mean = (loss / weights[timesteps - 1]).mean()
        assert not loss_1.isinf().any()
        assert not loss_2.isinf().any()
        assert not loss_mean.isinf()

        optimizer.zero_grad()
        loss_mean.backward()
        if gradient_clipping is not None:
            nn.utils.clip_grad_norm_(ndm.model.parameters(), gradient_clipping)
            nn.utils.clip_grad_norm_(ndm.model_F.parameters(), gradient_clipping)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss_mean.detach().item())
        global_step += 1

        if importance_sampling:
            for t, l in zip(timesteps, loss.detach()):
                idx = t - 1
                ptr = ndm.L_t_ptr[idx]
                ndm.L_t[idx, ptr] = l
                ndm.L_t_ptr[idx] = (ptr + 1) % ndm.importance_sampling_batch_size
                ndm.L_t_counts[idx] = min(
                    ndm.L_t_counts[idx] + 1, ndm.importance_sampling_batch_size
                )

    return global_step, epoch_losses
