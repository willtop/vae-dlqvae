import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import utils
from models.vqvae import VQVAE
from models.dlqvae import DLQVAE


parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--model", type=str, default="VQVAE")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=20000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=200)
parser.add_argument("--dataset",  type=str, default='CELEBA')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add in the dataset to the filename
model_filename = f"{args.model}-{args.dataset}-{timestamp}" 
print('Results will be saved in ./results/' + model_filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

if args.model == "VQVAE":
    # number of quantized vector per latent pixel
    n_embeddings = 64
    # depth of each latent pixel
    embedding_dim = 64
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_residual_layers, 
                  n_embeddings, embedding_dim, args.beta).to(device)
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

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():

    for i in tqdm(range(args.n_updates), desc="training epochs"):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            hyperparameters = args.__dict__
            utils.save_model_and_results(
                model, results, hyperparameters, model_filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


if __name__ == "__main__":
    train()
