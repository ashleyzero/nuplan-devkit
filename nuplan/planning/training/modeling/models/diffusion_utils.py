from typing import List, Tuple, Optional, cast

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import math
from nuplan.planning.training.modeling.models.diffusion_dynamics import DynamicsIntegrateOutput

class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0.0):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
    
class TransformerConcatLinear(nn.Module):

    def __init__(self, context_dim, pred_state_dim, tf_layer=3):
        super().__init__()
        # self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(pred_state_dim,2*context_dim,context_dim+3) # dim_in = |pred_state|
        # self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim), num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3) # dim_out = |pred_state|
        #self.linear = nn.Linear(128,2)


    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

class SampleSelector():
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
    
    def select_best(self, samples: List[DynamicsIntegrateOutput], route_coords: List[torch.tensor]) -> List[int]:
        assert len(samples) == self.num_samples
        batch_size = samples[0].poses.shape[0]
        best_indices = []
        for i in range(batch_size):
            scores = torch.zeros(self.num_samples).to(samples[0].poses.device)
            for j in range(self.num_samples):
                # poses, velocities, accelerations, jerks = samples[j]
                if route_coords[i].numel() > 0:
                    min_distance_to_route = torch.norm(route_coords[i] - samples[j].poses[i, 7, :2], dim=-1).min().item()
                    scores[j] -= max(min_distance_to_route - 2.0, 0.0)
                scores[j] -= samples[j].jerks[i, :, 0].abs().sum()
            # print(scores)
            best_indices.append(torch.argmax(scores).item())
        return best_indices

class Constraints():
    def __init__(
        self,
        max_distance_to_route: float,
        max_accleration: float,
        max_angular_acceleration: float
    ):
        self.max_distance_to_route = max_distance_to_route
        self.max_accleration = max_accleration
        self.max_angular_acceleration = max_angular_acceleration

    def compute_g_route(self, poses: torch.tensor, route_coords: List[torch.tensor]) -> torch.tensor:
        assert poses.shape[0] == len(route_coords)
        batch_size = poses.shape[0]
        g_route = torch.zeros(batch_size, poses.shape[1]).to(poses.device)
        for i in range(batch_size):
            if route_coords[i].numel() > 0:
                g_route[i] = (torch.norm(
                    route_coords[i].unsqueeze(1) - poses[i, :, :2].unsqueeze(0), dim=-1
                ).min(dim=0)[0] - self.max_distance_to_route).clamp(min=0.0)
        return g_route
    
    def compute_g_kinematic(self, accelerations: torch.tensor) -> torch.tensor:
        assert accelerations.shape[-1] == 2, f"Got accelerations dim: {accelerations.shape[-1]} , expected 2"
        g_kinematic = (accelerations[..., 0].abs() - self.max_accleration).clamp(min=0.0) + \
            (accelerations[..., 1].abs() - self.max_angular_acceleration).clamp(min=0.0)
        return g_kinematic
    
    def compute_g(
        self, poses: torch.tensor, accelerations: torch.tensor, route_coords: List[torch.tensor]
    ) -> torch.tensor:
        return self.compute_g_route(poses, route_coords) + self.compute_g_kinematic(accelerations)

def compute_d(poses1: torch.tensor, poses2: torch.tensor) -> torch.tensor:
    assert poses1.shape == poses2.shape
    assert poses1.shape[-1] == 3, f"Got poese dim: {poses1.shape[-1]} , expected 3"
    return torch.norm(torch.stack([
        poses1[..., 0] - poses2[..., 0],
        poses1[..., 1] - poses2[..., 1],
        torch.cos(poses1[..., 2]) - torch.cos(poses2[..., 2]),
        torch.sin(poses1[..., 2]) - torch.sin(poses2[..., 2]),
    ], dim=-1), dim=-1)
