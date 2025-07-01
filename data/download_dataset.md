# Download Instructions: Dogs vs Cats Dataset

1. Go to: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data
2. Download `train.zip` from the page.
3. Unzip the file into this directory:
4. Run the provided `prepare_data.py` script (optional) to:
- Filter broken files
- Organize into:
  ```
  data/processed/
      ├── train/cats/
      ├── train/dogs/
      ├── val/cats/
      └── val/dogs/
  ```