# Multiclass Classification for Alzheimer Disease

This project provides a deep learning-based web application for Alzheimer stage detection from MRI scans.  
It classifies uploaded brain MRI images into one of four classes:

- Mild Demented
- Moderate Demented
- Non Demented
- Very Mild Demented

## Project Files

- `app2.py` – Streamlit app for prediction and PDF report export
- `model.h5` – Trained TensorFlow/Keras model used by the app
- `alzheimer_final.ipynb` – Notebook used for model development/training workflow
- `requirements.txt` – Python dependencies

## Requirements

- Python 3.8+ (recommended)
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

From the project root:

```bash
streamlit run app2.py
```

Then open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

## How It Works

1. Enter patient details (name, age, gender, contact number).
2. Upload an MRI image (`.jpg`, `.jpeg`, `.png`).
3. The app preprocesses the image (RGB conversion, resize to `176x176`, normalization).
4. The model predicts the Alzheimer stage.
5. Optionally export a PDF report with patient info and prediction result.

## Notes

- Keep `model.h5` in the project root so `app2.py` can load it correctly.
- This tool is intended for educational/research use and does not replace clinical diagnosis.
