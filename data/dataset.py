from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from transforms import get_train_transforms, get_val_transforms

def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=2):
    # Load train dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_train_transforms())
    
    # Load combined val+test dataset
    combined_dataset = datasets.ImageFolder(root=val_dir, transform=get_val_transforms())
    
    # Split into val and test (e.g., 80%-20%)
    val_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - val_size
    val_dataset, test_dataset = random_split(combined_dataset, [val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
