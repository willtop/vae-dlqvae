import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
import utils
from models.vae import *


parser = argparse.ArgumentParser()

"""
Hyperparameters
"""

parser.add_argument("--model", type=str, default="VanillaVAE")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--dataset",  type=str, default='CELEBA')
parser.add_argument("--test", action="store_true")


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add in the dataset to the filename
os.makedirs(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "results"), exist_ok=True)
    
model_filepath = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "results", 
                    f"{args.model}-{args.dataset}.pth")

"""
Load data and define batch data loaders
"""

(
    training_data, 
    validation_data, 
    training_loader, 
    validation_loader, 
    x_train_var
) = utils.load_data_and_data_loaders(args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

if args.model == "VanillaVAE":
    model = VanillaVAE(args.latent_dim).to(device)
else:
    n_embeddings_per_dim = 10
    # depth of each latent pixel
    embedding_dim = 5
    model = DLQVAE(latent_dim_encoder=args.latent,
                   latent_dim_quant=10,
                   levels_per_dim=2
                   ).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)



def train():
    model.train()
    for i in tqdm(range(1, args.n_epochs+1), desc="training epochs"):
        for x, _ in training_loader:
            x = x.to(device)
            optimizer.zero_grad()
            if args.model == "VanillaVAE":
                mu, log_var = model.encode(x)
                z_sampled = model.reparameterize(mu, log_var)
                x_hat = model.decode(z_sampled)
                assert list(x.shape) == list(x_hat.shape)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_hat, target=x)
                loss_latent = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = loss_reconstruct + 0.0005 * loss_latent
            else:
                (
                    x_hat,
                    quant_idxs,
                    latent_loss_quant,
                    latent_loss_commit
                ) = model(x)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_hat, target=x)
                loss_latent = latent_loss_quant * 0.1 + latent_loss_commit * 0.1
                loss = loss_reconstruct + loss_latent
            loss.backward()
            optimizer.step()

        if i % args.log_interval == 0:
            print(f"Update # {i}" + 
                  f"Recon Loss: {loss_reconstruct.item():.3f}" +
                  f"Latent Loss: {loss_latent.item():.3f}" +
                  f"Total Loss:{loss.item():.3f}")
            """
            save model and print values
            """
            hyperparameters = args.__dict__
            utils.save_model_and_parameters(model, hyperparameters, model_filepath, args)

            
        
def test():
    checkpoint = torch.load(model_filepath)['model']
    model.load_state_dict(checkpoint)
    print(f"{args.model} model loaded successfully from {model_filepath}")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(validation_loader):
            x = x.to(device)
            if args.model == "VanillaVAE":
                mu, log_var = model.encode(x)
                z_sampled = model.reparameterize(mu, log_var)
                x_hat =  model.decode(z_sampled)
                assert torch.equal(x.shape, x_hat.shape)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_hat, target=x)
                loss_KL = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = loss_reconstruct + 0.0005 * loss_KL
            else:
                (
                    x_hat,
                    quant_idxs,
                    latent_loss_quant,
                    latent_loss_commit
                ) = model(x)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_hat, target=x)
                loss_latent = latent_loss_quant * 0.1 + latent_loss_commit * 0.1
                loss = loss_reconstruct + loss_latent
            test_loss += loss.item()
            # visualize real and reconstructed images
            if i == 0:
                n = 8
                comparison = torch.cat([x[:n], x_hat[:n]]).cpu()
                save_image(comparison, f'results/orig-recon-{args.model}.png', nrow=n)

    test_loss /= len(validation_loader)
    print(f'{args.model} Test set loss: {test_loss:.4f}')
    return

def generate():
    with torch.no_grad():
        z_sampled = torch.randn(64, args.latent_dim).to(device)
        x_sampled = model.decode(z_sampled).cpu()
        save_image(x_sampled, f"results/gen-samples-{args.model}.png", nrow=8)
    return

if __name__ == "__main__":
    if not args.test:
        train()
    else:
        test()
        generate()

    print("Script finished successfully!")
