    # 3. visualization.py — All Plots & Grad-CAM
    # Generates visuals for both training progress and model interpretability.

    # Processes:
    # Training Metrics Visualization:
    # Line plots:
    # Epoch vs Loss (Train/Val)
    # Epoch vs Accuracy (Train/Val)
    # Save plots to outputs/plots/
    # Confusion Matrix:
    # Plot confusion matrix as a heatmap using seaborn
    # Save as .png
    # Grad-CAM:
    # Generate Grad-CAM for a given image
    # Overlay heatmap on original image
    # Save to outputs/gradcam/
    
    # Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torchsummary import summary
import torch
    
def plot_training_metrics(model_name):
    df = pd.read_csv(f'outputs/logs/{model_name}.csv')
    train_acc = df['Train Accuracy']
    train_loss = df['Train Loss']
    val_acc = df['Val Accuracy']
    val_loss = df['Val Loss']
    epoch = df['Epoch']
    time = df['Time']
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the lines on the first axes
    ax1.plot(epoch, train_acc, label='Train Accuracy', color='green')
    ax1.plot(epoch, train_loss, label='Train Loss', color='red')
    ax1.plot(epoch, val_acc, label='Val Accuracy', color='blue')
    ax1.plot(epoch, val_loss, label='Val Loss', color='orange')

    # Set labels and title for the first axes
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value (Accuracy/Loss)')
    ax1.set_title(f'Training Metrics of Model {model_name}')
    ax1.tick_params(axis='y')

    # Calculate x-axis limits with a gap
    # You can adjust this 'buffer_epochs' value for more or less gap
    buffer_epochs = 0.5
    x_min_buffered = min(epoch) - buffer_epochs
    x_max_buffered = max(epoch) + buffer_epochs

    # Explicitly set x-ticks for the primary axis
    ax1.set_xticks(epoch)
    # Set x-axis limits with the calculated gap
    ax1.set_xlim(x_min_buffered, x_max_buffered)
    # IMPORTANT: Remove ax1.margins(x=0) as we are manually setting xlim for the gap
    # ax1.margins(x=0)

    # Move the legend outside the plot area
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create a twin axes sharing the x-axis
    ax2 = ax1.twiny()

    # Crucial for alignment: Set ax2 ticks to be exactly the same locations as ax1 ticks
    ax2.set_xticks(ax1.get_xticks())
    # Set the new x-axis labels, formatted to two decimal places for consistency
    ax2.set_xticklabels([f'{t:.4f} sec' for t in time])
    ax2.set_xlabel('Time')
    ax2.tick_params(axis='x')

    # Apply the same x-axis limits (with gap) to the twin axis
    ax2.set_xlim(x_min_buffered, x_max_buffered)
    # IMPORTANT: Remove ax2.margins(x=0)
    # ax2.margins(x=0)

    # Adjust the plot area to make space for the legend on the right
    plt.tight_layout(rect=(0, 0,1, 1))

    # Display the plot (save to file)
    plt.savefig('outputs/plots/train_metrics.png')
    
def plot_confusion_matrix(cm,model_name):
    
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix of Model {model_name}')
    plt.savefig('outputs/plots/confusion_matrix.png')

def Grad_cam():
    pass

def plot_layers(model):
    if torch.cuda.is_available():
        model.cuda()
        print(summary(model, (3, 128, 128), device="cuda"))
    else:
        print(summary(model, (3, 128, 128), device="cpu"))