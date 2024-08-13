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
from models.building_blocks import FactorVAE_Discriminator


parser = argparse.ArgumentParser()

"""
Hyperparameters
"""

parser.add_argument("--model", type=str, default="vanillavae", 
                    choices=['vanillavae', 'factorvae', 'dlqvae'])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=60)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--learning_rate_discriminator", type=float, default=1e-5)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--dataset",  type=str, default='celeba', 
                    choices=['celeba', 'mpi3d_toy', 'mpi3d_toy_pairs', 'mpi3d_complex', 'mpi3d_complex_pairs'])
parser.add_argument("--test", action="store_true")


args = parser.parse_args()

if args.dataset == "celeba":
    args.img_size = 224
elif args.dataset.startswith("mpi3d"):
    args.img_size = 64
else:
    print("not supported!")
    exit(1)

# for factorVAE, due to the training of discriminator, has to have a pair of images
# returned from the dataloader, reflected in this code base by the dataset name
if args.model == "factorvae":
    assert args.dataset.endswith("pairs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

os.makedirs(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "results"), exist_ok=True)
model_filepath = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "results", 
                    f"{args.model}_{args.dataset}.pth")

"""
Load data and define batch data loaders
"""

(
    training_data, 
    validation_data, 
    training_loader, 
    validation_loader
) = utils.load_data_and_data_loaders(args)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

auxiliary_discriminator, optimizer_discriminator = None, None 
if args.model == "vanillavae":
    model = VAE(args.latent_dim, args.img_size).to(device)
elif args.model == "factorvae":
    model = FactorVAE(args.latent_dim, args.img_size).to(device)
    auxiliary_discriminator = FactorVAE_Discriminator(args.latent_dim).to(device)
else:
    model = DLQVAE(latent_dim_encoder=args.latent_dim,
                   latent_dim_quant=50,
                   levels_per_dim=4,
                   img_size=args.img_size).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
if auxiliary_discriminator:
    optimizer_discriminator = optim.Adam(auxiliary_discriminator.parameters(), lr=args.learning_rate_discriminator, amsgrad=True)


def train():
    model.train()
    for i in tqdm(range(1, args.n_epochs+1), desc="training epochs"):
        for x, x2 in tqdm(training_loader, desc="minibatches within an epoch"):
            x, x2 = x.to(device), x2.to(device)
            if args.model == "vanillavae":
                optimizer.zero_grad()
                mu, log_var = model.encode(x)
                z_sampled = model.reparametrize(mu, log_var)
                x_hat = model.decode(z_sampled)
                assert list(x.shape) == list(x_hat.shape)
                # compute losses
                loss_reconstruct = utils.reconstruction_loss(x_hat, x)
                loss_latent = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = loss_reconstruct + 0.02 * loss_latent
                loss.backward()
                optimizer.step()
            elif args.model == "factorvae":
                optimizer.zero_grad()
                loss_kl_weight = 0.06
                loss_gamma_weight = 5 # value used in the FactorVAE repo
                ### loss for VAE parameters update ###
                mu, log_var = model.encode(x)
                z_sampled = model.reparametrize(mu, log_var)
                x_hat = model.decode(z_sampled)
                assert list(x.shape) == list(x_hat.shape)
                # conventional VAE losses
                # for reconstruction loss, original repo uses cross entropy
                loss_reconstruct = utils.reconstruction_loss(x_hat, x)
                loss_latent = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                # total correlation VAE loss
                p_logits_discriminator = auxiliary_discriminator(z_sampled)
                # in one repo it's commented if using discriminator which computes sigmoid
                # the following loss would change correspondingly, resulting inferior performance
                loss_kc = torch.mean(p_logits_discriminator[:,0]-p_logits_discriminator[:,1])
                anneal_val_gamma = utils.linear_annealing(0, 1, i, int(args.n_epochs/2))
                loss = loss_reconstruct + \
                        loss_kl_weight * loss_latent + \
                            anneal_val_gamma * loss_gamma_weight * loss_kc
                # Previously had retain_graph for p_logits_discriminator
                # Now that I am re-running the discriminator on the detached latent to get p_logits_discriminator
                # There shouldn't be need for retain_graph
                loss.backward() 
                optimizer.step()

                ### loss for discriminator parameters update ###
                optimizer_discriminator.zero_grad() # this step also clears the above undesired gradients on the discriminator
                zeros = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                ones = torch.ones_like(zeros)
                # compute the original latent encoding and discriminator logits again
                # with detached latent features to disentangle it from the above computation graph
                p_logits_discriminator = auxiliary_discriminator(z_sampled.detach())
                mu_2, log_var_2 = model.encode(x2)
                z_sampled_2 = model.reparametrize(mu_2, log_var_2)
                z_sampled_2_permed = utils.permute_dims(z_sampled_2).detach()
                p_logits_permed_discriminator = auxiliary_discriminator(z_sampled_2_permed)
                loss_discriminator = 0.5*(F.cross_entropy(p_logits_discriminator, zeros)+
                                            F.cross_entropy(p_logits_permed_discriminator, ones))
                loss_discriminator.backward()
                optimizer_discriminator.step()
            else:
                optimizer.zero_grad()
                (
                    x_hat,
                    quant_idxs,
                    latent_loss_quant,
                    latent_loss_commit
                ) = model(x)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_hat, target=x)
                loss_latent = latent_loss_quant * 1 + latent_loss_commit * 1
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
            if args.model in ["vanillavae", "factorvae"]:
                mu, log_var = model.encode(x)
                z_sampled = model.reparametrize(mu, log_var)
                x_hat =  model.decode(z_sampled)
                assert list(x.shape)==list(x_hat.shape)
                # compute losses
                loss_reconstruct = F.mse_loss(input=x_hat, target=x)
                loss_KL = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = loss_reconstruct + loss_KL
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
                save_image(comparison, f'results/orig_recon_{args.model}_{args.dataset}.png', nrow=n)

    test_loss /= len(validation_loader)
    print(f'{args.model} Test set loss: {test_loss:.4f}')

    if args.model == "dlqvae":
        print("Inspecting trained DLQVAE codebook...")
        model.inspect_learned_codebook()

    return

def generate():
    with torch.no_grad():
        x_sampled = model.sample_random_latent(81, device).cpu()
        save_image(x_sampled, f"results/gen_samples_{args.model}_{args.dataset}.png", nrow=9)
        if args.model in ["factorvae", "dlqvae"]:
            n_base_imgs = 3
            x_sampled_all = []
            for i in range(n_base_imgs):
                rand_img_idx = np.random.randint(len(validation_data))
                rand_img, _ = validation_data[rand_img_idx]
                x_sampled, n_vals_traversed = model.sample_traversed_latent(rand_img, device)
                x_sampled_all.append(x_sampled.view(-1, n_vals_traversed, 3, args.img_size, args.img_size))
            x_sampled_all = torch.concat(x_sampled_all, dim=1)
            assert x_sampled_all.shape[1:] == (n_vals_traversed*n_base_imgs, 3, args.img_size, args.img_size)
            save_image(x_sampled_all.view(-1, 3, args.img_size, args.img_size).cpu(), 
                       f"results/gen_samples_traverseLatent_{args.model}_{args.dataset}.png", 
                       nrow=n_vals_traversed*n_base_imgs)
    return

if __name__ == "__main__":
    if not args.test:
        train()
    else:
        test()
        generate()

    print("Script finished successfully!")
