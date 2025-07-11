# Load dataset, choose model (base/custom)

# Loss: CrossEntropyLoss

# Optimizer: Adam

# Scheduler (optional): StepLR or ReduceLROnPlateau

# Track training/validation loss, accuracy

# Save best model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from utils.metrics_logger import Logger
from utils.metrics_logger import log_in
from utils.visualization import plot_training_metrics

from config import device

import wandb

# def training(model,train_loader,val_loader,num_epochs=20 ):
#     log_in('Inside training function')
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(),lr=1e-3, weight_decay=1e-4)
#     scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
#     train_accuracy=0
#     val_loss=val_acc=0
#     best_accuracy=0
    
#     wandb.init(project=f"dogs-vs-cats_{model.__class__.__name__}")

    
#     wandb.config.update({
#         "model": model.__class__.__name__,
#         "epochs": num_epochs,
#         "loss": "CrossEntropyLoss",
#         "optimizer": "Adam"
#     })

    
#     log = Logger(model.__class__.__name__)
#     for epoch in range(num_epochs):
#         train_loss = 0
#         total = 0
#         correct = 0
        
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             # Accuracy
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             train_loss += loss.item()
#             correct += (predicted == labels).sum().item()
            
#         scheduler.step()
#         train_accuracy = 100 * correct / total
#         val_acc,val_loss=val(model,val_loader,criterion)
#         scheduler.step(val_loss) 
        
#         wandb.log({
#         "epoch": epoch,
#         "train_loss": train_loss,
#         "train_accuracy": train_accuracy,
#         "val_loss": val_loss ,
#         "val_accuracy": val_acc,
#         "learning_rate": scheduler.get_last_lr()[0] 
#     })

        
#         log.log_metrics(epoch,train_loss,train_accuracy,val_loss,val_acc)
        
#         if val_acc > best_accuracy:#storing the model checkpoints
#             best_accuracy = val_acc
#             log.log_best_metrics(epoch,train_loss,train_accuracy,val_loss,val_acc)
#             torch.save(model.state_dict(), f"outputs/checkpoints/best_{model.__class__.__name__}_model.pth")
#     plot_training_metrics(model.__class__.__name__)
            

def training(model, train_loader, val_loader, num_epochs):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    log = Logger(model.__class__.__name__)
    wandb.init(project=f"dogs-vs-cats_classification",
               name=f"{model.__class__.__name__}", 
               reinit=True,
               config={
        "model": model.__class__.__name__,
        "epochs": num_epochs,
        "optimizer": "Adam",
        "loss_function": "CrossEntropy",
        "scheduler": scheduler.__class__.__name__ if scheduler else "None"
    })

    wandb.watch(model, log="all", log_freq=1000)
    train_accuracy=0
    val_loss=val_acc=0
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
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

            # Track accuracy and loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            train_loss += loss.item()
            correct += (predicted == labels).sum().item()

        # Average loss and accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validation
        val_acc, val_loss = val(model, val_loader, criterion)  # Your val function returns accuracy and loss

        # Step LR scheduler if ReduceLROnPlateau
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": scheduler.optimizer.param_groups[0]['lr']
        })

        # Log locally
        log.log_metrics(epoch, avg_train_loss, train_accuracy, val_loss, val_acc)

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            log.log_best_metrics(epoch, avg_train_loss, train_accuracy, val_loss, val_acc)
            torch.save(model.state_dict(), f"outputs/checkpoints/best_{model.__class__.__name__}_model.pth")

    # After all epochs
    plot_training_metrics(model.__class__.__name__)
    wandb.finish()
    
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


