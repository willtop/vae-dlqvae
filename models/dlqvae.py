
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder_DLQVAE
from models.quantizer import LatentQuantizer
from models.decoder import Decoder_DLQVAE


class DLQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings_per_dim, embedding_dim):
        super(DLQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder_DLQVAE(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = LatentQuantizer(
                levels = n_embeddings_per_dim,
                # avoid the need for internal linear projection layer
                dim = embedding_dim,
                codebook_dim = embedding_dim
            )
        # decode the discrete latent representation
        self.decoder = Decoder_DLQVAE(embedding_dim, h_dim, n_res_layers, res_h_dim)


    def forward(self, x):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        z_q, _, embedding_loss = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        # return a placeholder for perplexity
        return embedding_loss, x_hat, torch.tensor(-1)



    

