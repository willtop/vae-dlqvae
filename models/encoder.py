import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def construct_vae_encoder(latent_dim):
    in_dim = 3
    hidden_dims = [32, 64, 128, 256, 512]
    
    encoder_conv_list = []
    for h_dim in hidden_dims:
        encoder_conv_list.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=h_dim,
                            kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_dim = h_dim
    encoder_conv_lyrs = nn.Sequential(*encoder_conv_list)
    # throw in a pseudo image to see the convolution layers output size
    dummy_out = encoder_conv_lyrs(torch.rand(1, 3, 227, 227))
    conv_out_size = dummy_out.shape[2]
    # add in flattened fully connected layers
    encoder_fc_mu = nn.Linear(hidden_dims[-1] * conv_out_size**2, latent_dim)
    encoder_fc_var = nn.Linear(hidden_dims[-1] * conv_out_size**2, latent_dim)
    return encoder_conv_lyrs, encoder_fc_mu, encoder_fc_var, hidden_dims[-1], conv_out_size
