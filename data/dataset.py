# dataset.py
from torchvision import datasets
from torch.utils.data import DataLoader
from transforms import get_train_transforms, get_val_transforms

def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=2):# sending the data dir for transformations, and loading via Dataloader
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_train_transforms())#generic data loader
    val_dataset = datasets.ImageFolder(root=val_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)# generates a iterable with batch of images 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
