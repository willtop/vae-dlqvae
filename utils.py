import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
#from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np

# newly added loader for celebA
def load_celeba():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224))
    ])
    train = datasets.CelebA("data", 
                            split='train', 
                            target_type='identity',
                            transform=data_transforms,
                            download=True)
    
    val = datasets.CelebA("data", 
                          split='valid', 
                          target_type='identity',
                          transform=data_transforms,
                          download=True)
    return train, val

def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


# def load_block():
#     data_folder_path = os.getcwd()
#     data_file_path = data_folder_path + \
#         '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

#     train = BlockDataset(data_file_path, train=True,
#                          transform=transforms.Compose([
#                              transforms.ToTensor(),
#                              transforms.Normalize(
#                                  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                          ]))

#     val = BlockDataset(data_file_path, train=False,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize(
#                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                        ]))
#     return train, val

# def load_latent_block():
#     data_folder_path = os.getcwd()
#     data_file_path = data_folder_path + \
#         '/data/latent_e_indices.npy'

#     train = LatentBlockDataset(data_file_path, train=True,
#                          transform=None)

#     val = LatentBlockDataset(data_file_path, train=False,
#                        transform=None)
#     return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'CELEBA':
        training_data, validation_data = load_celeba()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        # CelebA doesn't have readily obtainable attribute for getting images
        training_imgs = torch.stack([training_data[i][0] for i in range(len(training_data))],axis=0)
        x_train_var = np.var(training_imgs.numpy())

    # elif dataset == 'BLOCK':
    #     training_data, validation_data = load_block()
    #     training_loader, validation_loader = data_loaders(
    #         training_data, validation_data, batch_size)
    #     x_train_var = np.var(training_data.data / 255.0)

    # elif dataset == 'LATENT_BLOCK':
    #     training_data, validation_data = load_latent_block()
    #     training_loader, validation_loader = data_loaders(
    #         training_data, validation_data, batch_size)
    #     x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and CELEBA datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, filename):
    SAVE_MODEL_PATH = os.path.join(os.getcwd(), 'results')
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               os.path.join(SAVE_MODEL_PATH, filename + '.pth'))
