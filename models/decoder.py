import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_vae_decoder(hidden_dims, latent_dim, encoder_conv_out_size):
    decoder_conv_list = []
    decoder_fc = nn.Linear(latent_dim, hidden_dims[-1] * encoder_conv_out_size**2)
    # reverse the convolutional layer channels, don't alter the outside list
    hidden_dims = hidden_dims.copy()
    hidden_dims.reverse()
    for i in range(len(hidden_dims) - 1):
        decoder_conv_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[i], 
                    out_channels=hidden_dims[i+1],
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU())
        )
    # final convolution layer
    decoder_conv_list.append( 
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())
        )    
    decoder_conv_lyrs = nn.Sequential(*decoder_conv_list)
    
    return decoder_fc, decoder_conv_lyrs

if __name__ == "__main__":
    decoder_fc , decoder_conv_lyrs = construct_vae_decoder(latent_dim=256, encoder_conv_out_size=14)
