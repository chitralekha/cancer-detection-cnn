# Histopathologic Cancer Detection

## Project Overview

This project develops a Convolutional Neural Network (CNN) model to detect metastatic cancer in histopathologic image patches. The dataset consists of over 220,000 microscopic tissue images labeled as cancerous or non-cancerous. Automated detection of cancerous tissue can assist pathologists by reducing diagnostic time and improving diagnostic accuracy.

## Dataset

- Approximately 220,000 RGB images, each sized 96×96 pixels.
- Binary labels:
  - 1 = cancerous tissue
  - 0 = non-cancerous tissue
- Images provided as individual `.tif` files, with labels provided in a CSV file.
- Dataset organized with separate folders for training and testing images.

## Methodology

- Data loaded from Google Drive and preprocessed using TensorFlow's `ImageDataGenerator` with augmentation techniques such as rotation, zoom, and flipping to improve robustness and handle class imbalance.
- A CNN model was built with multiple convolutional layers, max pooling, dropout, and fully connected dense layers using Keras Sequential API.
- The model was trained for 10 epochs with an 80-20 training-validation split.
- Model performance was evaluated using accuracy, confusion matrix, precision, recall, and F1-score.

## Results

- Achieved approximately **91% validation accuracy**.
- Confusion matrix and classification report show balanced precision and recall, though some false negatives remain, indicating room for improved sensitivity.
- Training and validation loss and accuracy curves indicate effective learning without significant overfitting.

## Challenges and Learnings

- Handling large datasets efficiently through batch processing and data generators.
- Addressing class imbalance through data augmentation.
- Balancing model complexity with computational resources.
- Importance of monitoring training to avoid overfitting.

## Future Work

- Experimenting with deeper and pretrained models like ResNet or EfficientNet through transfer learning.
- Incorporating class weighting or focal loss to better manage class imbalance.
- Applying hyperparameter tuning techniques to optimize model performance.
- Using early stopping and learning rate scheduling to enhance training efficiency.
- Exploring ensemble methods for improved robustness.

## How to Use

1. **Clone this repository:**

    ```bash
    git clone https://github.com/your-username/histopathologic-cancer-detection.git
    cd histopathologic-cancer-detection
    ```

2. **Download the dataset:**

    Download the **Histopathologic Cancer Detection** dataset from the official Kaggle competition page:  
    [https://www.kaggle.com/c/histopathologic-cancer-detection/data](https://www.kaggle.com/c/histopathologic-cancer-detection/data)  
    You will need to create a free Kaggle account if you don’t have one.

3. **Upload the dataset ZIP file to Google Drive:**

    Upload the downloaded `histopathologic-cancer-detection.zip` file to a folder in your Google Drive.  
    Note the folder path, and update the `zip_path` variable in the notebook accordingly (default is `/content/drive/MyDrive/histopathologic-cancer-detection.zip`).

4. **Run the notebook:**

    Open the Jupyter notebook (`Histopathologic_Cancer_Detection.ipynb`) in Google Colab or your local environment.  
    Run the cells sequentially to:  
    - Mount Google Drive and unzip the dataset (if not already extracted).  
    - Load and preprocess the data with augmentation.  
    - Build, train, and evaluate the CNN model.

5. **Save and load the trained model:**

    The notebook includes code to save the trained model to your Google Drive for reuse without retraining. You can also load this saved model later to perform inference or further analysis.

## Files

- `Histopathologic_Cancer_Detection.ipynb` — Jupyter notebook containing all code and markdown explanations.  
- `histopathologic-cancer-detection.zip` — Dataset archive (not included due to size; download separately from Kaggle).  
- `README.md` — This file.
