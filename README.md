# Multiclass Classification — Alzheimer Disease (MRI)

This project builds a multi-class CNN model to classify brain MRI scans into four Alzheimer’s disease stages and ships a Streamlit web app for end‑to‑end inference and reporting.

## Project Highlights
- **4-class MRI classification**: Non‑Demented, Very Mild Demented, Mild Demented, Moderate Demented.
- **Streamlit web app** with patient data capture and **PDF report export**.
- **Validation accuracy exceeds 95%** in notebook runs (best val_acc **0.9761**).
- **TensorFlow/Keras CNN** with separable conv blocks, batch normalization, and dropout.

## Results (from `alzheimer_final.ipynb`)
The notebook logs show:
- **Training accuracy**: best **0.9839** (~98.39%).
- **Validation accuracy**: best **0.9761** (~97.61%).
- **Test accuracy**: **0.9141** (~91.41%) on a held‑out test split.

> Note: Metrics depend on data splits, preprocessing, and training configuration. These values are from a single notebook run for research/demo purposes and should not be interpreted as clinical validation.

## Dataset
The model is trained on the Kaggle dataset:
- **Dataset**: `marcopinamonti/alzheimer-mri-4-classes-dataset`
- **Classes**:
  - `NonDemented`
  - `VeryMildDemented`
  - `MildDemented`
  - `ModerateDemented`

## Model Overview
The CNN architecture (summary of the notebook model):
- Input: **176×176 RGB**
- Conv2D + MaxPooling
- SeparableConv2D blocks with BatchNorm + MaxPooling
- Dropout regularization
- Dense layers (512 → 128 → 64) with BatchNorm + Dropout
- Output: **4‑class softmax**

## Training Pipeline
Key steps in the notebook:
1. **Download dataset** from Kaggle API.
2. **Image preprocessing**: resize to 176×176, rescale to [0, 1].
3. **Data augmentation**: brightness, zoom, horizontal flip.
4. **SMOTE oversampling** to balance classes.
5. **Train/val/test split**:
   - 80% train / 20% test
   - 20% of train used for validation (≈64% train / 16% val / 20% test)
6. Train for up to 50 epochs and evaluate on test set.

## Web App (Streamlit)
The Streamlit app (`app2.py`) lets users:
- Enter patient details (name, age, gender, contact).
- Upload an MRI scan image (jpg/png).
- Get a predicted class label.
- Export a **PDF report** with patient details, scan, and predictions.

### Run the App
```bash
pip install -r requirements.txt
streamlit run app2.py
```

The app expects `model.h5` in the project root. This repository includes `model.h5`; you can also regenerate it by running the notebook and saving the trained model.

## Training / Reproducing Results
The notebook `alzheimer_final.ipynb` contains the full training pipeline.

### Requirements for Training (not included in `requirements.txt`)
`requirements.txt` targets **runtime dependencies for the Streamlit app**. The notebook uses additional **training-time** packages such as:
- pandas, numpy, opencv-python, matplotlib
- imbalanced-learn, kaggle

Install them before running the notebook:
```bash
pip install pandas numpy opencv-python matplotlib imbalanced-learn kaggle
```

### Kaggle API Setup
1. Create a Kaggle API token.
2. Place `kaggle.json` in `~/.kaggle/` (or set `KAGGLE_CONFIG_DIR`).
3. Run the dataset download cells in the notebook.

## Repository Structure
```
.
├── alzheimer_final.ipynb   # Training + evaluation notebook
├── README.md               # Project documentation
├── app2.py                 # Streamlit inference app
├── model.h5                # Trained Keras model
├── requirements.txt        # App runtime dependencies
└── LICENSE
```

## Notes & Limitations
- This is a research/demo project and **not a clinical diagnostic tool**.
- Performance can vary based on data distribution and preprocessing.
- The current app validates inputs but assumes reasonable MRI image quality.

## License
See [LICENSE](LICENSE).
