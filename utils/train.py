# Load dataset, choose model (base/custom)

# Loss: CrossEntropyLoss

# Optimizer: Adam

# Scheduler (optional): StepLR or ReduceLROnPlateau

# Track training/validation loss, accuracy

# Save best model

import torch
from data.dataset import get_dataloaders
from models.custom_cnn import get_models
from torch.nn.functional import binary_cross_entropy as bce
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.metrics_logger import Logger
from utils.metrics_logger import log_in
from utils.evaluate import evaluate_model
from utils.visualization import plot_training_metrics

from config import device


def training(model,train_loader,val_loader,num_epochs=20 ):
    log_in('Inside training function')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train_accuracy=0
    val_loss=val_acc=0
    best_accuracy=0
    
    log = Logger(model.__class__.__name__)
    for epoch in range(num_epochs):
        train_loss = 0
        total = 0
        correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            train_loss += loss.item()
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        train_accuracy = 100 * correct / total
        val_acc,val_loss=val(model,val_loader,criterion)
        scheduler.step(val_loss) 
        
        log.log_metrics(epoch,train_loss,train_accuracy,val_loss,val_acc)
        
        if val_acc > best_accuracy:#storing the model checkpoints
            best_accuracy = val_acc
            log.log_best_metrics(epoch,train_loss,train_accuracy,val_loss,val_acc)
            torch.save(model.state_dict(), f"outputs/checkpoints/best_{model.__class__.__name__}_model.pth")
    plot_training_metrics(model.__class__.__name__)
            
def val(model, val_loader, criterion):
    log_in('Inside Validation function')
    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()

    val_accuracy = val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)
    

    return  val_accuracy,val_loss


