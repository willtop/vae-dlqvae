from __future__ import annotations
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, int32
from torch.nn import Module
from torch.optim import Optimizer
import numpy as np
from einops import pack, rearrange, unpack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FactorVAE_Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(FactorVAE_Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_units = 500
        self.net = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_units),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.LeakyReLU(0.2, True),
            # outputs two logits for D(z) and 1-D(z) respectively
            # so later in VAE and discriminator losses (which contain log)
            # can directly compute without taking log
            nn.Linear(self.hidden_units, 2)
        )

    def forward(self, z):
        return self.net(z)

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


"""
Code adapted  and modified on top of
https://github.com/lucidrains/vector-quantize-pytorch
Disentanglement via Latent Quantization
 - https://arxiv.org/abs/2305.18378
"""


class LatentQuantizer(Module):
    def __init__(
        self,
        latent_dim,
        levels_per_dim
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.levels_per_dim = levels_per_dim

        # ensure zero is in the middle and start is always -0.5
        values_per_latent = [
            torch.linspace(-0.5, 0.5, self.levels_per_dim) if self.levels_per_dim % 2 == 1 \
                                else torch.arange(self.levels_per_dim) / self.levels_per_dim - 0.5
            for _ in range(self.latent_dim)
        ]

        # Add the values per latent into the model parameters for optimization
        self.values_per_latent = nn.ParameterList(
            [nn.Parameter(values) for values in values_per_latent]
        )
            

    def compute_latent_quant_loss(self, z: Tensor, zhat: Tensor) -> Tensor:
        """Computes the quantization loss."""
        return F.mse_loss(zhat.detach(), z, reduction="mean")

    def compute_latent_commit_loss(self, z: Tensor, zhat: Tensor) -> Tensor:
        """Computes the commitment loss."""
        return F.mse_loss(z.detach(), zhat, reduction="mean")

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z.
        The quantization is done by measuring the distance between the input and the codebook values per latent dimension
        and returning the index of the closest codebook value.
        """

        def distance(x, y):
            return torch.abs(x - y)

        quant_idxs = torch.stack(
            [
                torch.argmin(distance(z[:, i].view(-1,1), self.values_per_latent[i]), dim=1)
                    for i in range(self.latent_dim)
            ],
            dim=-1,
        )
        z_quant = torch.stack(
            [
                self.values_per_latent[i][quant_idxs[:, i]]
                for i in range(self.latent_dim)
            ],
            dim=-1,
        )
        
        return z_quant, quant_idxs


    def forward(self, z: Tensor) -> Tensor:
        assert (z.shape[-1] == self.latent_dim), f"{z.shape[-1]} VS {self.latent_dim}"
        z_quant, quant_idxs = self.quantize(z)

        # compute the two-part latent loss here, before cutting off 
        # the gradients to the latent code
        loss_quant = self.compute_latent_quant_loss(z, z_quant)
        loss_commit = self.compute_latent_commit_loss(z, z_quant)

        # This code is brought outsize the quantize(), later to here
        # preserve the gradients on z for reconstruction loss via the 
        # straight-through gradient estimator
        # however this would cut off the gradient of the z_quant
        z_quant_for_recon = z + (z_quant - z).detach()

        return z_quant_for_recon, quant_idxs, loss_quant, loss_commit
