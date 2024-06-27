import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np

# newly added loader for celebA
def load_celeba():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224)
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

    elif dataset == 'CELEBA':
        training_data, validation_data = load_celeba()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and CELEBA datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader



def save_model_and_parameters(model, hyperparameters, filepath, args):
    results_to_save = {
        'model': model.state_dict(),
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, filepath)
    print(f"{args.model} model saved successfully at: ", filepath)
    return
