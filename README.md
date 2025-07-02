# Dogs_vs_Cats_classification

# 📄 Project Documentation: Dog vs Cat Image Classifier (PyTorch + Web App)

## 📁 Project Structure (Top-Level Overview)

```
dog-cat-classifier/
├── data/
├── models/
├── utils/
├── outputs/
├── webapp/
├── main.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 📂 Folder & File Structure (Detailed)

### 1. `data/` - Dataset Handling & Preprocessing

* **Purpose**: To load, preprocess, and augment the dataset for training and testing.
* **Files**:

  * `dataset.py`: Custom PyTorch Dataset class that loads Dog/Cat images, assigns labels, and supports train/test splits.
  * `transforms.py`: Centralized file for applying image transformations (resize, normalization, augmentation).
  * `download_dataset.md`: Instructions for downloading and setting up the Kaggle Dogs vs Cats dataset manually.
  * `__init__.py`: Marks this directory as a Python package.

### 2. `models/` - Neural Network Architectures

* **Purpose**: Contains model definitions and supporting building blocks.
* **Files**:

  * `base_cnn.py`: A baseline CNN with 3-4 convolutional layers, max-pooling, and dropout.
  * `attention_block.py`: Contains the implementation of attention mechanisms such as SE Block or CBAM.
  * `custom_cnn.py`: Advanced CNN integrating the attention block, and optional residual connections.
  * `__init__.py`: Package marker.

### 3. `utils/` - Supporting Tools for Training & Evaluation

* **Purpose**: Contains reusable components and helper functions.
* **Files**:

  * `train.py`: Implements the training loop with model checkpointing, learning rate scheduling, and progress tracking.
  * `evaluate.py`: Implements the evaluation pipeline that calculates metrics like Accuracy, Precision, Recall, and F1.
  * `visualization.py`: Functions to generate plots (loss vs epoch, accuracy vs epoch), confusion matrix, and Grad-CAM visualizations.
  * `metrics_logger.py`: Logs metrics to CSV/JSON and manages experiment tracking.
  * `__init__.py`: Makes `utils` importable as a module.

### 4. `outputs/` - Generated Output Artifacts

* **Purpose**: Stores results generated during training and evaluation.
* **Structure**:

  * `checkpoints/`: Stores best-performing model weights.
  * `logs/`: Contains training logs and experiment notes.
  * `plots/`: Loss and accuracy plots, confusion matrix, etc.
  * `gradcam/`: Stores Grad-CAM images generated for test samples.

### 5. `webapp/` - Streamlit Interface for Image Upload & Prediction

* **Purpose**: Provides a simple GUI for users to upload an image and view model predictions with Grad-CAM visualization.
* **Files**:

  * `app.py`: Streamlit app entry point.
  * `utils.py`: Helper functions to load the model, preprocess images, and render Grad-CAM.
  * `examples/`: A few sample test images for demonstration.
  * `README.md`: Deployment and usage instructions for the web interface.

### 6. Top-Level Files

* **`main.py`**: Entry point for model training; uses components from `data/`, `models/`, and `utils/`.
* **`evaluate.py`**: Script to evaluate a saved model and generate metrics and visualizations.
* **`requirements.txt`**: Python dependencies for the entire project.
* **`README.md`**: Project overview, setup instructions, architecture details, metrics summary, and demo link.

---

## ✨ Key Project Features

* Custom CNN built from scratch
* Attention modules (SE/CBAM) integrated
* Grad-CAM visual explanations
* Web interface using Streamlit
* Detailed training & evaluation metrics
* Model size, inference time, and comparative experiments

---
