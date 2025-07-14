import os
import sys
import uuid
import shutil
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import pandas as pd
import subprocess

sys.path.append(os.path.abspath('.'))
from models.custom_cnn import get_models
from utils.visualization import Grad_cam
from utils.evaluate import test_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FIX: Placed UPLOAD_FOLDER inside 'static' to make files servable
import sys
import os

# Add the root directory (one level up) to the Python path

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Automatically copy plot images to static so HTML can access them
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PLOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
STATIC_PLOT_DIR = os.path.join(BASE_DIR, "static", "outputs", "plots")
os.makedirs(STATIC_PLOT_DIR, exist_ok=True)

if os.path.exists(ORIGINAL_PLOT_DIR):
    for filename in os.listdir(ORIGINAL_PLOT_DIR):
        src = os.path.join(ORIGINAL_PLOT_DIR, filename)
        dst = os.path.join(STATIC_PLOT_DIR, filename)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
#ensure checkpoints exist
os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "checkpoints"), exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attention_model, base_model = get_models()

ATTN_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_custom_attention_cnn_model.pth")
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_custom_basic_cnn_model.pth")

base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
attention_model.load_state_dict(torch.load(ATTN_MODEL_PATH, map_location=device))

base_model.eval()
attention_model.eval()

# Load training metrics
BASIC_CSV = os.path.join(PROJECT_ROOT, "outputs", "logs", "custom_basic_cnn_best.csv")
ATTN_CSV = os.path.join(PROJECT_ROOT, "outputs", "logs", "custom_attention_cnn_best.csv")

df_basic = pd.read_csv(BASIC_CSV, names=['Epoch', 'Time', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
df_attn = pd.read_csv(ATTN_CSV, names=['Epoch', 'Time', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])

last_base = df_basic.iloc[-1]
last_attn = df_attn.iloc[-1]

base_metrics = {
    "train_acc": round(last_base["Train Accuracy"], 2),
    "train_loss": round(last_base["Train Loss"], 4),
    "val_acc": round(last_base["Val Accuracy"], 2),
    "val_loss": round(last_base["Val Loss"], 4),
}

attn_metrics = {
    "train_acc": round(last_attn["Train Accuracy"], 2),
    "train_loss": round(last_attn["Train Loss"], 4),
    "val_acc": round(last_attn["Val Accuracy"], 2),
    "val_loss": round(last_attn["Val Loss"], 4),
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file or not file.filename:
            return render_template("image_classifier.html", uploaded=False)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")

        base_label, base_conf = test_image(base_model, img)
        attn_label, attn_conf = test_image(attention_model, img)

        base_result = {"label": "Dog" if base_label.item() == 1 else "Cat", "confidence": round(base_conf.item() * 100, 2)}
        attn_result = {"label": "Dog" if attn_label.item() == 1 else "Cat", "confidence": round(attn_conf.item() * 100, 2)}

        gradcam_base_filename = f"gradcam_base_{uuid.uuid4().hex}.png"
        gradcam_attn_filename = f"gradcam_attention_{uuid.uuid4().hex}.png"

        gradcam_base_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_base_filename)
        gradcam_attn_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_attn_filename)

        Grad_cam(model=base_model, orig_image=img, save_path=gradcam_base_path)
        Grad_cam(model=attention_model, orig_image=img, save_path=gradcam_attn_path)

        # FIX: Generate correct relative paths for url_for
        image_rel_path = os.path.join("uploads", filename)
        gradcam_base_rel_path = os.path.join("gradcam", gradcam_base_filename)
        gradcam_attn_rel_path = os.path.join("gradcam", gradcam_attn_filename)

        return render_template("image_classifier.html",
                               uploaded=True,
                               # FIX: Use url_for to generate correct static URLs
                               image_src=url_for('static', filename=image_rel_path),
                               base_result=base_result,
                               attention_result=attn_result,
                               gradcam_base=url_for('static', filename=gradcam_base_rel_path),
                               gradcam_attention=url_for('static', filename=gradcam_attn_rel_path),
                               train_base_acc=base_metrics["train_acc"],
                               train_base_loss=base_metrics["train_loss"],
                               val_base_acc=base_metrics["val_acc"],
                               val_base_loss=base_metrics["val_loss"],
                               train_attention_acc=attn_metrics["train_acc"],
                               train_attention_loss=attn_metrics["train_loss"],
                               val_attention_acc=attn_metrics["val_acc"],
                               val_attention_loss=attn_metrics["val_loss"]
                               )

    return render_template("image_classifier.html",
                           uploaded=False,
                           train_base_acc=base_metrics["train_acc"],
                           train_base_loss=base_metrics["train_loss"],
                           val_base_acc=base_metrics["val_acc"],
                           val_base_loss=base_metrics["val_loss"],
                           train_attention_acc=attn_metrics["train_acc"],
                           train_attention_loss=attn_metrics["train_loss"],
                           val_attention_acc=attn_metrics["val_acc"],
                           val_attention_loss=attn_metrics["val_loss"])


if __name__ == "__main__":
    app.run(debug=True)