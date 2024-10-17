import random
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, DataLoader
import os
import numpy as np
import h5py


class Custom_Dataset(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return np.shape(self.imgs)[0]

    def __getitem__(self, index):
        return (self.transforms(self.imgs[index]), torch.tensor(0))
    
# for FactorVAE training
class Custom_Dataset_Pairs(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs = imgs
        self.transforms = transforms
    
    def __len__(self):
        return np.shape(self.imgs)[0]
    
    def __getitem__(self, index):
        img1 = self.imgs[index]
        img2_id = random.choice(range(np.shape(self.imgs)[0]))
        img2 = self.imgs[img2_id]
        return (self.transforms(img1), self.transforms(img2))

# newly added loader for celebA
def load_celeba(args):
    assert args.img_size == 224
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

def load_cifar(args):
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

def load_mpi3d(args, ds_type, whether_pairs=False):
    assert args.img_size == 64
    data_transforms = transforms.Compose([
         transforms.ToTensor()
    ])
    if ds_type == "toy":
        datafile_path = os.path.join("data", "mpi3d_toy.npz")
    elif ds_type == "complex":
        datafile_path = os.path.join("data", "real3d_complicated_shapes_ordered.npz")
    else:
        print("MPI3D unsupported type: ", ds_type)
        exit(1)
    print(f"Loading mpi3d {ds_type} data from {datafile_path}...")

    mpi3d_data = np.load(datafile_path)['images']
    n_imgs = mpi3d_data.shape[0]
    if ds_type == "toy":
        assert n_imgs == 1_036_800
        n_train_imgs = 1_000_000
    elif ds_type == "complex":
        assert n_imgs == 460_800
        n_train_imgs = 400_000
    else:
        print("MPI3D unsupported type: ", ds_type)
        exit(1)

    if whether_pairs:
        mpi3d_dataset = Custom_Dataset_Pairs(mpi3d_data, data_transforms)
    else:
        mpi3d_dataset = Custom_Dataset(mpi3d_data, data_transforms)
    
    print("[MPI3D] Getting meta splits.....")
    # get meta split indices
    perm = np.arange(n_imgs)
    np.random.shuffle(perm)
    # use a certain number of samples for training the encoder
    # note: while this is the same number fo samples as in Diversified metaML setup
    # the exact samples aren't guaranteed to be identical due to shuffled indices 
    meta_train_idxs, meta_valid_idxs = \
                perm[:n_train_imgs], perm[n_train_imgs:]
    
    mpi3d_meta_train, mpi3d_meta_valid = \
                Subset(mpi3d_dataset, meta_train_idxs), \
                Subset(mpi3d_dataset, meta_valid_idxs)
    
    return mpi3d_meta_train, mpi3d_meta_valid

def load_shapes3d(args, whether_pairs=False):
    assert args.img_size == 64
    data_transforms = transforms.Compose([
         transforms.ToTensor()
    ])
    datafile_path = os.path.join("data", "shapes3d.h5")

    print(f"Loading shapes3d data from {datafile_path}...")

    shapes3d_data = h5py.File(datafile_path, 'r')
    shapes3d_imgs = shapes3d_data['images'][()]
    n_imgs = shapes3d_imgs.shape[0]
    assert n_imgs == 480_000
    n_train_imgs = 400_000

    if whether_pairs:
        shapes3d_dataset = Custom_Dataset_Pairs(shapes3d_imgs, data_transforms)
    else:
        shapes3d_dataset = Custom_Dataset(shapes3d_imgs, data_transforms)
    
    print("[Shapes3D] Getting meta splits.....")
    # get meta split indices
    perm = np.arange(n_imgs)
    np.random.shuffle(perm)
    # use a certain number of samples for training the encoder
    # note: while this is the same number fo samples as in Diversified metaML setup
    # the exact samples aren't guaranteed to be identical due to shuffled indices 
    meta_train_idxs, meta_valid_idxs = \
                perm[:n_train_imgs], perm[n_train_imgs:]
    
    shapes3d_meta_train, shapes3d_meta_valid = \
                Subset(shapes3d_dataset, meta_train_idxs), \
                Subset(shapes3d_dataset, meta_valid_idxs)
    
    return shapes3d_meta_train, shapes3d_meta_valid


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


def load_data_and_data_loaders(args):
    if args.dataset == 'cifar10':
        training_data, validation_data = load_cifar(args)
    elif args.dataset == 'celeba':
        training_data, validation_data = load_celeba(args)
    elif args.dataset == 'mpi3d_toy':
        training_data, validation_data = load_mpi3d(args, ds_type="toy")
    elif args.dataset == 'mpi3d_toy_pairs':
        training_data, validation_data = load_mpi3d(args, ds_type="toy", whether_pairs=True)
    elif args.dataset == 'mpi3d_complex':
        training_data, validation_data = load_mpi3d(args, ds_type="complex")
    elif args.dataset == 'mpi3d_complex_pairs':
        training_data, validation_data = load_mpi3d(args, ds_type="complex", whether_pairs=True)
    elif args.dataset == "shapes3d":
        training_data, validation_data = load_shapes3d(args)    
    elif args.dataset == "shapes3d_pairs":
        training_data, validation_data = load_shapes3d(args, whether_pairs=True)
    else:
        raise ValueError(f'Invalid dataset name: {args.dataset}.')
    training_loader, validation_loader = data_loaders(
            training_data, validation_data, args.batch_size)

    return training_data, validation_data, training_loader, validation_loader

def reconstruction_loss(x_hat, x):
    return F.binary_cross_entropy(input=x_hat, target=x)
    #return F.mse_loss(input=x_hat, target=x)

def linear_annealing(init_val, final_val, step, total_steps):
    delta_val = (final_val - init_val)/total_steps
    annealed_val = init_val + delta_val * step
    return annealed_val


def permute_dims(z):
    assert z.dim() == 2

    batch_size, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(batch_size).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

def save_model_and_parameters(model, hyperparameters, filepath, args):
    results_to_save = {
        'model': model.state_dict(),
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, filepath)
    print(f"{args.model} model saved successfully at: ", filepath)
    return
