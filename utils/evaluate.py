import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json
from utils.visualization import plot_confusion_matrix
from utils.metrics_logger import log_in
from config import device
from data.dataset import load_test_image


def evaluate_model(model,test_loader):
    #load best checkpoint of model
    log_in('Inside evaluate_model function')
    load_model = torch.load(f'outputs/checkpoints/best_{model.__class__.__name__}_model.pth')
    model.load_state_dict(load_model)
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_labels = []
    model.to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)  

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()# convert numpy array to list for JSON serialization
    }

    with open(f'outputs/logs/metrics_{model.__class__.__name__}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    ##Plot confusion matrix via visualization.py
    plot_confusion_matrix(cm,model.__class__.__name__)
    
    return None
     
def test_image(model,img):
    log_in('Inside test_image function')
    model.eval()
    model.to(device)
    img_tensor,_ = load_test_image(img)
    model.eval()
    # prediction=0
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1)
        prediction = prediction.item()

    
    return int(prediction)