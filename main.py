import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import argparse
from tqdm import tqdm
import utils
from models.vae import *


parser = argparse.ArgumentParser()

"""
Hyperparameters
"""

parser.add_argument("--model", type=str, default="VanillaVAE")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--latent_space", type=int, default=256)
parser.add_argument("--n_updates", type=int, default=20000)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=200)
parser.add_argument("--dataset",  type=str, default='CELEBA')
parser.add_argument("--test", action="store_true")


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add in the dataset to the filename
model_filename = f"results/{args.model}-{args.dataset}.pth" 
print("Model location: ", model_filename)

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
    model = DLQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers,
                   n_embeddings_per_dim, embedding_dim).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)



def train():
    model.train()
    for i in tqdm(range(1, args.n_updates+1), desc="training epochs"):
        for batch_idx, (x, _) in enumerate(training_loader):
            x = x.to(device)
            optimizer.zero_grad()
            if args.model == "VanillaVAE":
                mu, log_var = model.encode(x)
                z_sampled = model.reparameterize(mu, log_var)
                x_reconstructed =  model.decode(z_sampled)
                assert torch.equal(x.shape, x_reconstructed.shape)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_reconstructed, target=x)
                loss_KL = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = loss_reconstruct + 0.0005 * loss_KL
            else:
                pass
            loss.backward()
            optimizer.step()

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            hyperparameters = args.__dict__
            utils.save_model_and_parameters(model, hyperparameters, model_filename, args)

            print(f"Update # {i}" + 
                  f"Recon Loss: {loss_reconstruct.item():.3f}" +
                  f"KL Loss: {loss_KL.item():.3f}" +
                  f"Total Loss:{loss.item():.3f}")
        
def test():
    checkpoint = torch.load(model_filename)['model']
    model.load_state_dict(checkpoint)
    print(f"{args.model} model loaded successfully from {model_filename}")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(validation_loader):
            x = x.to(device)
            if args.model == "VanillaVAE":
                mu, log_var = model.encode(x)
                z_sampled = model.reparameterize(mu, log_var)
                x_reconstructed =  model.decode(z_sampled)
                assert torch.equal(x.shape, x_reconstructed.shape)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_reconstructed, target=x)
                loss_KL = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = loss_reconstruct + 0.0005 * loss_KL
                test_loss += loss.item()
                # visualize real and reconstructed images
                if i == 0:
                    n = 8
                    comparison = torch.cat([x[:n], x_reconstructed[:n]]).cpu()
                    save_image(comparison,
                               f'results/orig_recon_{args.model}.png', nrow=n)

    test_loss /= len(validation_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return

def generate():
    with torch.no_grad():
        sample = torch.randn(64, args.latent_dim).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample,
                    f"results/generated_samples_{args.model}.png", nrow=8)
    return

if __name__ == "__main__":
    if not args.test:
        train()
    else:
        test()
        generate()

    print("Script finished successfully!")
