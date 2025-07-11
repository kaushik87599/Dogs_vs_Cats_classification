from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from data.transforms import get_train_transforms, get_val_transforms
from utils.metrics_logger import log_in
from PIL import Image
import torchvision.transforms as T
import numpy as np
from config import device
import torch

def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=2):
    # Load train dataset
    log_in('Inside get_dataloaders')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_train_transforms())
    log_in(f'train_dataset classes {train_dataset.class_to_idx}')
    # Load combined val+test dataset
    combined_dataset = datasets.ImageFolder(root=val_dir, transform=get_val_transforms())
    log_in(f'combined_dataset classes {combined_dataset.class_to_idx}')
    log_in('Finished Transformations')
    

    # Split into val and test (e.g., 80%-20%)
    val_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - val_size
    val_dataset, test_dataset = random_split(combined_dataset, [val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log_in('Finsihed Dataloaders')
    return train_loader, val_loader, test_loader

def load_dir(dir,batch_size=32, num_workers=2):
    log_in('-'*50+'\nInside load_dir\n')
    dir_data = datasets.ImageFolder(root=dir, transform=get_val_transforms())
    log_in(f'dir_data classes {dir_data.class_to_idx}')
    dir_data = DataLoader(dir_data, batch_size=batch_size, shuffle=False)
    return dir_data


def load_test_image(image_path):
    
    
    if isinstance(image_path, str):
        img =  Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        img = image_path
    else:
        raise TypeError("Input must be a file path or PIL.Image.Image instance")
    orig_image = np.array(img)
    transf = get_val_transforms()
    img_tensor = transf(img)  # Apply transform to get a tensor
    image_tensor = torch.unsqueeze(img_tensor, 0)  # type: ignore # Add batch dimension
    image_tensor = image_tensor.to(device)
    return image_tensor, orig_image

# def load_test_image(path):
#     log_in('-'*50+'\nInside load_test_image\n')
#     test_image = datasets.ImageFolder(root=path, transform=get_val_transforms())
#     log_in(f'test_image classes {test_image.class_to_idx}')
#     test_image = DataLoader(test_image, batch_size=1, shuffle=False)
#     return test_image
