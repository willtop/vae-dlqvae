import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_vae_decoder(conv_channels, latent_dim, encoder_conv_out_size):
    decoder_conv_list = []
    decoder_fc = nn.Linear(latent_dim, conv_channels[-1] * encoder_conv_out_size**2)
    # reverse the convolutional layer channels, don't alter the outside list
    conv_channels = conv_channels.copy()
    conv_channels.reverse()
    for i in range(len(conv_channels) - 1):
        decoder_conv_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=conv_channels[i], 
                    out_channels=conv_channels[i+1],
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                    output_padding=1),
                nn.BatchNorm2d(conv_channels[i+1]),
                nn.LeakyReLU())
        )
    # final convolution layer
    decoder_conv_list.append( 
        nn.Sequential(
            nn.ConvTranspose2d(conv_channels[-1],
                               conv_channels[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(conv_channels[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=conv_channels[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())
        )    
    decoder_conv_lyrs = nn.Sequential(*decoder_conv_list)
    
    return decoder_fc, decoder_conv_lyrs
