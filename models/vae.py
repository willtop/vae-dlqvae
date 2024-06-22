
import torch
import torch.nn as nn
import numpy as np
from .encoder import construct_vae_encoder
from .quantizer import LatentQuantizer
from .decoder import construct_vae_decoder

class VanillaVAE(nn.Module):
    def __init__(self, latent_dim):
        super(VanillaVAE, self).__init__()
        self.conv_channels = [32, 64, 128, 256, 512]
        self.latent_dim = latent_dim
        # construct encoder module
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu,
            self.encoder_fc_var, 
            self.encoder_conv_out_size
        ) = construct_vae_encoder(self.conv_channels, self.latent_dim)
        # construct decoder module
        (
            self.decoder_fc,
            self.decoder_conv_lyrs 
        ) = construct_vae_decoder(self.conv_channels, self.latent_dim, 
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
        z = z.view(-1, self.conv_channels[-1], self.encoder_conv_out_size, self.encoder_conv_out_size)
        x_hat = self.decoder_conv_lyrs(z)
        return x_hat


class DLQVAE(nn.Module):
    def __init__(self, latent_dim_encoder, latent_dim_quant, levels_per_dim):
        super(DLQVAE, self).__init__()
        self.conv_channels = [32, 64, 128, 256, 512]
        self.latent_dim_encoder = latent_dim_encoder
        self.latent_dim_quant = latent_dim_quant
        # number of levels per dimension in the latent space to be quantized
        self.levels_per_dim = levels_per_dim
        # construct encoder module
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu,
            _, # don't need the variation prediction layer
            self.encoder_conv_out_size
        ) = construct_vae_encoder(self.conv_channels, self.latent_dim_encoder)
        self.fc_encoder_to_quant = nn.Linear(self.latent_dim_encoder, self.latent_dim_quant)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = LatentQuantizer(
                latent_dim = self.latent_dim_quant,                
                levels_per_dim = self.levels_per_dim
            )
        self.fc_quant_to_decoder = nn.Linear(self.latent_dim_quant, self.latent_dim_encoder)
        # construct decoder module
        (
            self.decoder_fc,
            self.decoder_conv_lyrs 
        ) = construct_vae_decoder(self.conv_channels, self.latent_dim_encoder, 
                                  encoder_conv_out_size=self.encoder_conv_out_size)
        


    def forward(self, x):
        z = self.encoder_conv_lyrs(x)
        z = torch.flatten(z, start_dim=1)
        z = self.encoder_fc_mu(z)
        z = self.fc_encoder_to_quant(z)
        (
            z_q, 
            quant_idxs, 
            latent_loss_quant, 
            latent_loss_commit
        ) = self.vector_quantization(z)
        z_q = self.fc_quant_to_decoder(z_q)
        z_q = self.decoder_fc(z_q)
        z_q = z_q.view(-1, self.conv_channels[-1], self.encoder_conv_out_size, self.encoder_conv_out_size)
        x_hat = self.decoder_conv_lyrs(z_q)

        return x_hat, quant_idxs, latent_loss_quant, latent_loss_commit
