import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def construct_vae_encoder(conv_params, latent_dim):
    in_dim = 3 
    encoder_conv_list = []
    for (n_chnl, kn_size, strd, pad, _) in conv_params:
        encoder_conv_list.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_dim, 
                          out_channels=n_chnl,
                          kernel_size=kn_size, 
                          stride=strd, 
                          padding=pad),
                nn.BatchNorm2d(n_chnl),
                nn.LeakyReLU())
        )
        in_dim = n_chnl
    encoder_conv_lyrs = nn.Sequential(*encoder_conv_list)
    # throw in a pseudo 224X224 image (same as preprocessed celebA) 
    # to see the convolution layers output size
    dummy_out = encoder_conv_lyrs(torch.rand(1, 3, 224, 224))
    conv_out_size = dummy_out.shape[2]
    # add in flattened fully connected layers
    encoder_fc_mu = nn.Sequential(
        nn.Linear(conv_params[-1][0] * conv_out_size**2, 4096),
        nn.LeakyReLU(),
        nn.Linear(4096, latent_dim)
    )
    encoder_fc_var = nn.Sequential(
        nn.Linear(conv_params[-1][0] * conv_out_size**2, 4096),
        nn.LeakyReLU(),
        nn.Linear(4096, latent_dim)
    )
    return encoder_conv_lyrs, encoder_fc_mu, encoder_fc_var, conv_out_size

def construct_vae_decoder(conv_params, latent_dim, encoder_conv_out_size):
    decoder_conv_list = []
    decoder_fc = nn.Sequential(
        nn.Linear(latent_dim, 4096),
        nn.LeakyReLU(),
        nn.Linear(4096, conv_params[-1][0] * encoder_conv_out_size**2)
    )
    # reverse the order of convolutional layer specs, copy so don't alter the outside list
    conv_params = conv_params.copy()
    conv_params.reverse()
    for i in range(len(conv_params) - 1):
        decoder_conv_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=conv_params[i][0], 
                    out_channels=conv_params[i+1][0],
                    kernel_size=conv_params[i][1], 
                    stride=conv_params[i][2], 
                    padding=conv_params[i][3],
                    output_padding=conv_params[i][4]),
                nn.BatchNorm2d(conv_params[i+1][0]),
                nn.LeakyReLU())
        )
    # final convolution layer
    decoder_conv_list.append( 
        nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=conv_params[-1][0],
                out_channels=3,
                kernel_size=conv_params[-1][1],
                stride=conv_params[-1][2],
                padding=conv_params[-1][3],
                output_padding=conv_params[-1][4]),
            nn.Sigmoid())
        )    
    decoder_conv_lyrs = nn.Sequential(*decoder_conv_list)
    
    return decoder_fc, decoder_conv_lyrs

if __name__ == "__main__":
    input_dim = 224
    conv_params = [(96, 11, 4, 2, 1), 
                    (256, 5, 2, 2, 0), 
                    (384, 3, 2, 1, 1), 
                    (384, 3, 2, 1, 1),
                    (256, 3, 1, 1, 0)]
    
    print("encoder hidden dimensions")
    for _, ks, st, pd, _ in conv_params:
        output_dim = np.floor((input_dim + 2*pd - 1*(ks-1)-1)/st +1)
        print(output_dim)
        input_dim = output_dim

    conv_params.reverse()

    print("decoder hidden dimensions")
    for _, ks, st, pd, pdo in conv_params:
        output_dim = (input_dim-1)*st - 2*pd + 1*(ks-1) + pdo + 1
        print(output_dim)
        input_dim = output_dim 