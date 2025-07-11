import os
import sys
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
# from torchvision import transforms
import torch
import math
sys.path.append(os.path.abspath('.')) 

from models.custom_cnn import get_models
from utils.visualization import Grad_cam
from utils.evaluate import test_image  # your helper to get prediction + confidence
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER


# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attention_model,base_model  = get_models()

BASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "best_custom_basic_cnn_model.pth")
ATTN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "checkpoints", "best_custom_attention_cnn_model.pth")

base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
attention_model.load_state_dict(torch.load(ATTN_MODEL_PATH, map_location=device))


base_model.eval()
attention_model.eval()

import pandas as pd

# Paths to metric CSVs
BASIC_CSV = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs", "custom_basic_cnn_best.csv")
ATTN_CSV = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs", "custom_attention_cnn_best.csv")

df_basic = pd.read_csv(BASIC_CSV)
df_attn = pd.read_csv(ATTN_CSV)

# Get final epoch metrics
last_base = df_basic.iloc[-1]
last_attn = df_attn.iloc[-1]

base_metrics = {
    "train_acc": round(last_base["train_accuracy"], 2),
    "train_loss": round(last_base["train_loss"], 4),
    "val_acc": round(last_base["val_accuracy"], 2),
    "val_loss": round(last_base["val_loss"], 4),
}

attn_metrics = {
    "train_acc": round(last_attn["train_accuracy"], 2),
    "train_loss": round(last_attn["train_loss"], 4),
    "val_acc": round(last_attn["val_accuracy"], 2),
    "val_loss": round(last_attn["val_loss"], 4),
}

# Transform for uploaded image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("image_classifier.html", uploaded=False)

        if not file.filename:
            return render_template("image_classifier.html", uploaded=False)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert("RGB")
        

        # Predict with both models
        base_label, base_conf = test_image(base_model, img)
        attn_label, attn_conf = test_image(attention_model, img)

        base_result = {"label": base_label.item(), "confidence": round(base_conf.item() * 100, 2)}
        attn_result = {"label": attn_label.item(), "confidence": round(attn_conf.item() * 100, 2)}

        # Generate Grad-CAMs
        gradcam_base_filename = f"gradcam_base_{uuid.uuid4().hex}.png"
        gradcam_attn_filename = f"gradcam_attention_{uuid.uuid4().hex}.png"

        gradcam_base_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_base_filename)
        gradcam_attn_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_attn_filename)

        Grad_cam(model=base_model, orig_image=img, save_path=gradcam_base_path)
        Grad_cam(model=attention_model, orig_image=img, save_path=gradcam_attn_path)

        return render_template("image_classifier.html",
                               uploaded=True,
                               image_src=f"/{filepath}",
                               base_result=base_result,
                               attention_result=attn_result,
                               gradcam_base=f"/{gradcam_base_path}",
                               gradcam_attention=f"/{gradcam_attn_path}"
                               )

    return render_template("image_classifier.html", uploaded=False)

if __name__ == "__main__":
    app.run(debug=True)
