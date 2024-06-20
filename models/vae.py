
import torch
import torch.nn as nn
import numpy as np
from .encoder import construct_vae_encoder
from .quantizer import LatentQuantizer
from .decoder import construct_vae_decoder

class VanillaVAE(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(VanillaVAE, self).__init__()
        # construct encoder module
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu,
            self.encoder_fc_var, 
            self.last_hidden_channels, 
            self.encoder_conv_out_size
        ) = construct_vae_encoder(self.latent_dim)
        # construct decoder module
        (
            self.decoder_fc,
            self.decoder_conv_lyrs 
        ) = construct_vae_decoder(self.latent_dim, 
                                  encoder_conv_out_size=self.encoder_conv_out_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std)
        return noise * std + mu

    def encode(self, x):
        z = self.encoder_conv_lyrs(x)
        z = torch.flatten(z, start_dim=1)
        mu = self.encoder_fc_mu(z)
        log_var = self.encoder_fc_var(z)
        return mu, log_var
    
    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, self.last_hidden_channels, self.encoder_conv_out_size, self.encoder_conv_out_size)
        x = self.decoder_conv_lyrs(z)
        return x


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