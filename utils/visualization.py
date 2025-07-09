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
import cv2      
from utils.metrics_logger import log_in
import torch.nn.functional as F



def plot_training_metrics(model_name):
    log_in('Inside plot_training_metrics')
    df = pd.read_csv(f'outputs/logs/{model_name}.csv',names=['Epoch', 'Time', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
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
    ax2.tick_params(axis='x',rotation=90)

    # Apply the same x-axis limits (with gap) to the twin axis
    ax2.set_xlim(x_min_buffered, x_max_buffered)
    # IMPORTANT: Remove ax2.margins(x=0)
    # ax2.margins(x=0)

    # Adjust the plot area to make space for the legend on the right
    plt.tight_layout(rect=(0, 0,1, 1))

    # Display the plot (save to file)
    plt.savefig(f'outputs/plots/train_metrics_of_{model_name}.png')
    
def plot_confusion_matrix(cm,model_name):
    log_in('Inside plot_confusion_matrix')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix of Model {model_name}')
    plt.savefig(f'outputs/plots/confusion_matrix_{model_name}.png')

def get_last_conv_layer(model):
    # model.base_cnn is nn.Sequential of Layer modules
    log_in('Inside get_last_conv_layer')
    for layer in reversed(model.base_cnn):
        if hasattr(layer, 'conv') and isinstance(layer.conv, torch.nn.Conv2d):
            return layer.conv
    raise ValueError("No Conv2d layer found in model.base_cnn")

def Grad_cam(model, image_tensor, orig_image=None, class_idx=None):
    log_in('Inside Grad_cam')
    save_path=f'outputs/gradcam/gradcam_{model.__class__.__name__}.png'
    
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')
        model.to('cuda')
    model.eval()
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output.detach())
        print("Forward activation shape:", output.shape)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
        print("Backward gradient shape:", grad_out[0].shape)
        
    conv_layer = get_last_conv_layer(model)
    forward_handle = conv_layer.register_forward_hook(forward_hook)
    backward_handle = conv_layer.register_full_backward_hook(backward_hook)
    


    # 1. Forward pass
    output = model(image_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    # 2. Zero grads and backward pass for the target class
    model.zero_grad()
    target = output[0, class_idx]
    target.backward()

    # 3. Get activations and gradients
    activations = feature_maps[0]   # [1, C, H, W]
    grads = gradients[0]            # [1, C, H, W]

    # 4. Compute weights and Grad-CAM
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    grad_cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    grad_cam = torch.relu(grad_cam)
    grad_cam = F.interpolate(grad_cam, size=(128, 128), mode='bilinear', align_corners=False)
    grad_cam = grad_cam.squeeze().cpu().numpy()
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)  # Normalize to

    # 5. Overlay heatmap on original image
    if orig_image is not None:
        # orig_image: numpy array, shape [H, W, 3], range [0,255] or [0,1]
        grad_cam_uint8 = np.uint8(255 * grad_cam)
        grad_cam_uint8 = np.squeeze(grad_cam_uint8)
        if grad_cam_uint8.ndim != 2:
            raise ValueError("grad_cam_uint8 must be a 2D array for applyColorMap")
        heatmap = cv2.applyColorMap(grad_cam_uint8, cv2.COLORMAP_JET)
        if orig_image.max() <= 1.0:
            orig_image = (orig_image * 255).astype(np.uint8)
        # Ensure both images are uint8 and same shape
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        if orig_image.shape != heatmap.shape:
            heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
        # Set opacity to 50% for the heatmap
        overlay = cv2.addWeighted(orig_image, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    else:
        plt.imsave(save_path, grad_cam, cmap='jet')

    # 6. Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Clear for next use
    feature_maps.clear()
    gradients.clear()

def plot_layers(model):
    log_in('Inside plot_layers function')
    if torch.cuda.is_available():
        model.cuda()
        print(summary(model, (3, 128, 128), device="cuda"))
    else:
        print(summary(model, (3, 128, 128), device="cpu"))