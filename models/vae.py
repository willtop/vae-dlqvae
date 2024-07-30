
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .vae_constructor import construct_vae_encoder, construct_vae_decoder
from .building_blocks import LatentQuantizer

class VanillaVAE(nn.Module):
    def __init__(self, latent_dim):
        super(VanillaVAE, self).__init__()
        self.conv_params = [(32, 11, 4, 2, 1), 
                            (32, 5, 2, 2, 0), 
                            (64, 3, 2, 1, 1), 
                            (64, 3, 2, 1, 1),
                            (128, 3, 1, 1, 0)]
        self.latent_dim = latent_dim
        # construct encoder module
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu,
            self.encoder_fc_var, 
            self.encoder_conv_out_size
        ) = construct_vae_encoder(self.conv_params, 
                                  self.latent_dim,
                                  256)
        print(f"Constructed VanillaVAE, with output size after encoder convolution layers: {self.conv_params[-1][0]}X{self.encoder_conv_out_size}X{self.encoder_conv_out_size}")
        # construct decoder module
        (
            self.decoder_fc,
            self.decoder_conv_lyrs 
        ) = construct_vae_decoder(self.conv_params, 
                                  self.latent_dim, 
                                  256,
                                  encoder_conv_out_size=self.encoder_conv_out_size)

    def reparametrize(self, mu, log_var):
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
        z = z.view(-1, self.conv_params[-1][0], self.encoder_conv_out_size, self.encoder_conv_out_size)
        x_hat = self.decoder_conv_lyrs(z)
        return x_hat
    
    def sample_random_latent(self, n_samples, device):
        self.eval()
        z_sampled = torch.randn(n_samples, self.latent_dim).to(device)
        x_sampled = self.decode(z_sampled)
        return x_sampled
    

class FactorVAE(nn.Module):
    def __init__(self, latent_dim):
        super(FactorVAE, self).__init__()
        self.conv_params = [(32, 11, 4, 2, 1), 
                            (32, 5, 2, 2, 0), 
                            (64, 3, 2, 1, 1), 
                            (64, 3, 2, 1, 1),
                            (128, 3, 1, 1, 0)]
        self.latent_dim = latent_dim
        # construct encoder module
        (
            self.encoder_conv_lyrs, 
            self.encoder_fc_mu,
            self.encoder_fc_var, 
            self.encoder_conv_out_size
        ) = construct_vae_encoder(self.conv_params, 
                                  self.latent_dim, 
                                  256)
        print(f"Constructed FactorVAE, with output size after encoder convolution layers: {self.conv_params[-1][0]}X{self.encoder_conv_out_size}X{self.encoder_conv_out_size}")
        # construct decoder module
        (
            self.decoder_fc,
            self.decoder_conv_lyrs 
        ) = construct_vae_decoder(self.conv_params, 
                                  self.latent_dim, 
                                  256,
                                  encoder_conv_out_size=self.encoder_conv_out_size)

    def reparametrize(self, mu, log_var):
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
        z = z.view(-1, self.conv_params[-1][0], self.encoder_conv_out_size, self.encoder_conv_out_size)
        x_hat = self.decoder_conv_lyrs(z)
        return x_hat
    
    def sample_random_latent(self, n_samples, device):
        self.eval()
        z_sampled = torch.randn(n_samples, self.latent_dim).to(device)
        x_sampled = self.decode(z_sampled)
        return x_sampled


class DLQVAE(nn.Module):
    def __init__(self, latent_dim_encoder, latent_dim_quant, levels_per_dim):
        super(DLQVAE, self).__init__()
        self.conv_params = [(64, 3, 1, 0, 0), 
                            (128, 3, 2, 0, 1), 
                            (256, 5, 2, 1, 1), 
                            (256, 5, 3, 1, 0),
                            (128, 5, 3, 1, 0),
                            (64, 3, 2, 1, 1),
                            (64, 3, 2, 1, 0)]
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
        ) = construct_vae_encoder(self.conv_params, self.latent_dim_encoder, 256)
        print(f"Constructed DLQVAE, with output size after encoder convolution layers: {self.conv_params[-1][0]}X{self.encoder_conv_out_size}X{self.encoder_conv_out_size}")
        if self.latent_dim_quant != self.latent_dim_encoder:
            self.fc_encoder_to_quant = nn.Linear(self.latent_dim_encoder, self.latent_dim_quant)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantizer = LatentQuantizer(
                latent_dim = self.latent_dim_quant,                
                levels_per_dim = self.levels_per_dim
            )
        if self.latent_dim_quant != self.latent_dim_encoder:
            self.fc_quant_to_decoder = nn.Linear(self.latent_dim_quant, self.latent_dim_encoder)
        # construct decoder module
        (
            self.decoder_fc,
            self.decoder_conv_lyrs 
        ) = construct_vae_decoder(self.conv_params, self.latent_dim_encoder, 256,
                                  encoder_conv_out_size=self.encoder_conv_out_size)
        
    def encode(self, x):
        z = self.encoder_conv_lyrs(x)
        z = torch.flatten(z, start_dim=1)
        z = self.encoder_fc_mu(z)
        if self.latent_dim_quant != self.latent_dim_encoder:
            z = self.fc_encoder_to_quant(z)
        return z
        
    def decode(self, z):
        if self.latent_dim_quant != self.latent_dim_encoder:
            z = self.fc_quant_to_decoder(z)
        z = self.decoder_fc(z)
        z = z.view(-1, self.conv_params[-1][0], self.encoder_conv_out_size, self.encoder_conv_out_size)
        x_hat = self.decoder_conv_lyrs(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)        
        (
            z_q, 
            quant_idxs, 
            latent_loss_quant, 
            latent_loss_commit
        ) = self.vector_quantizer(z)
        x_hat = self.decode(z_q)

        return x_hat, quant_idxs, latent_loss_quant, latent_loss_commit

    def sample_random_latent(self, n_samples, device):
        self.eval()
        z_sampled = torch.stack([
            self.vector_quantizer.values_per_latent[i][
                torch.randint(high=self.levels_per_dim, size=(n_samples,))]
                    for i in range(self.latent_dim_quant)
        ],dim=-1).to(device)
        assert z_sampled.shape == (n_samples, self.latent_dim_quant)
        x_sampled = self.decode(z_sampled)
        return x_sampled
    
    def sample_traversed_latent(self, n_latent_dims_to_traverse, device):
        assert n_latent_dims_to_traverse <= self.latent_dim_quant
        self.eval()
        latent_dims_traversed = np.random.choice(np.arange(self.latent_dim_quant), 
                                                 size=n_latent_dims_to_traverse,
                                                 replace=False)
        z_sampled_base = torch.tensor([
            self.vector_quantizer.values_per_latent[i][
                torch.randint(high=self.levels_per_dim, size=(1,))]
                    for i in range(self.latent_dim_quant)])
        z_sampled_all = []
        for latent_dim in latent_dims_traversed:
            for latent_val in self.vector_quantizer.values_per_latent[latent_dim]:
                z_sampled = z_sampled_base.clone()
                # latent_val is a tensor with a single value, nonetheless can directly assign it 
                # to a 1-D tensor
                z_sampled[latent_dim] = latent_val
                z_sampled_all.append(z_sampled)
        z_sampled_all = torch.stack(z_sampled_all, dim=0).to(device)
        assert z_sampled_all.shape == (n_latent_dims_to_traverse * self.levels_per_dim, self.latent_dim_quant)
        x_sampled = self.decode(z_sampled_all)
        return x_sampled


    def inspect_learned_codebook(self):
        for i in range(self.latent_dim_quant):
            vals = self.vector_quantizer.values_per_latent[i].data
            print(vals)
        return


