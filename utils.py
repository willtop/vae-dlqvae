import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
import os
import numpy as np

class MPI3D(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return np.shape(self.imgs)[0]

    def __getitem__(self, index):
        return (self.transforms(self.imgs[index]), torch.tensor(0))

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

def load_mpi3d():
    data_transforms = transforms.Compose([
         transforms.ToTensor(),
    ])
    datafile_path = os.path.join("data", "real3d_complicated_shapes_ordered.npz")
    print(f"Loading mpi3d data from {datafile_path}...")

    mpi3d_data = np.load(datafile_path)['images']
    n_imgs = mpi3d_data.shape[0]
    assert n_imgs == 460800

    mpi3d_dataset = MPI3D(mpi3d_data, data_transforms)
    
    print("[MPI3D] Getting meta splits.....")
    # get meta split indices
    perm = np.arange(n_imgs)
    np.random.shuffle(perm)
    meta_train_idxs, meta_valid_idxs = \
                perm[:400000], perm[400000:]
    
    mpi3d_meta_train, mpi3d_meta_valid = \
                Subset(mpi3d_dataset, meta_train_idxs), \
                Subset(mpi3d_dataset, meta_valid_idxs)
    
    return mpi3d_meta_train, mpi3d_meta_valid


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
    if dataset == 'cifar10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

    elif dataset == 'celeba':
        training_data, validation_data = load_celeba()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        
    elif dataset == 'mpi3d':
        training_data, validation_data = load_mpi3d()
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
