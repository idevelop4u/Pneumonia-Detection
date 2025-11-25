Pneumonia Detection System

An AI-powered application that detects Pneumonia from chest X-ray images using Deep Learning.

Features

Deep Learning Model: Utilizes a ResNet34 architecture trained with FastAI.

High Accuracy: Uses Transfer Learning to achieve high performance on medical imaging.

Web Interface: Includes a Streamlit app for easy user interaction.

Data Augmentation: Implements rotation, zoom, and lighting adjustments to improve generalization.

Project Structure

Pneumonia-Detection/
├── data/                   # Dataset Folder
├── models/                 # Saved Models (.pkl)
├── src/                    # Source Code
│   ├── config.py           # Settings
│   ├── data_loader.py      # Data Processing
│   ├── model.py            # CNN Architecture
│   └── train_utils.py      # Training Loop
├── app.py                  # Streamlit Web App
├── main.py                 # Training Entry Point
├── requirements.txt        # Dependencies
└── statement.md            # Problem Statement


Installation

Clone the repository:

git clone <your-repo-link>
cd Pneumonia-Detection


Install dependencies:

pip install -r requirements.txt


Setup Dataset:

Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

Extract it so the folder structure is: data/chest_xray/train, data/chest_xray/test.

1. Train the Model

To train the model from scratch and generate the pneumonia_classifier.pkl file:

python main.py


2. Run the Web App

To start the user interface:

streamlit run app.py


This will open a local web server (usually at http://localhost:8501) where you can upload images for testing.

Technologies Used

Python 3.10+

FastAI / PyTorch (Deep Learning)

Pandas (Data Manipulation)

Streamlit (Frontend UI)
