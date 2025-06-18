# brain-tumor-detection
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/akshayaxv/brain-tumor-detection)

This repository contains a project for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN) based on the VGG16 architecture. The model is built using TensorFlow and Keras.

## Project Overview

The primary goal of this project is to classify MRI images of brains into two categories: those with tumors ('yes') and those without tumors ('no'). This is achieved by leveraging transfer learning with a pre-trained VGG16 model, fine-tuning it with additional custom layers for the specific task of brain tumor detection.

## Features

*   Image preprocessing including resizing and normalization.
*   Data augmentation to increase the diversity of the training set.
*   Transfer learning using the VGG16 model pre-trained on ImageNet.
*   Custom classification head with Dense, Dropout, and AveragePooling layers.
*   Training and evaluation scripts.
*   Visualization of training progress (loss and accuracy).
*   Model evaluation using classification reports and confusion matrices.

## Dataset

The model expects the dataset to be structured in a directory, for example, `./Desktop/DataFlair/brain_tumor_dataset/`, with two subdirectories:
*   `yes`: Containing MRI images with brain tumors.
*   `no`: Containing MRI images without brain tumors.

The script automatically loads images from these directories, resizes them to 224x224 pixels, and preprocesses them.

## Model Architecture

The model uses the VGG16 architecture as its base, with its convolutional layers frozen (not trainable) to leverage pre-trained weights from ImageNet. The following layers are added on top of the VGG16 base:

1.  **VGG16 Base Model** ( `include_top=False` )
2.  **AveragePooling2D** (pool_size=(4, 4))
3.  **Flatten**
4.  **Dense** (64 units, ReLU activation)
5.  **Dropout** (0.5 rate)
6.  **Dense** (2 units, Softmax activation for binary classification)

The model is compiled using the Adam optimizer with a learning rate of `1e-3` and `binary_crossentropy` as the loss function. The primary metric tracked is `accuracy`.

## Dependencies

The project relies on the following Python libraries:

*   TensorFlow (Keras API)
*   Scikit-learn
*   OpenCV-Python (`cv2`)
*   imutils
*   NumPy
*   Matplotlib

You can install these dependencies using pip:
```bash
pip install tensorflow scikit-learn opencv-python imutils numpy matplotlib
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/akshayaxv/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2.  **Prepare your dataset:**
    Ensure your dataset is structured as described in the "Dataset" section. Update the `path` variable in `brain_tumor_detection.py` or `brain_tumor_detection.ipynb` to point to your dataset directory if it's different.
    Currently, the path is set to:
    ```python
    path = "./Desktop/DataFlair/brain_tumor_dataset"
    ```

3.  **Run the script or notebook:**

    *   **Using the Python script:**
        ```bash
        python brain_tumor_detection.py
        ```
    *   **Using the Jupyter Notebook:**
        Open `brain_tumor_detection.ipynb` in Jupyter Notebook or JupyterLab and run the cells sequentially.

## Training and Evaluation

The script performs the following steps:
1.  **Loads and preprocesses images:** Reads images, resizes them to 224x224, normalizes pixel values to the [0, 1] range, and performs one-hot encoding on labels.
2.  **Splits data:** Divides the dataset into training (90%) and testing (10%) sets.
3.  **Data augmentation:** Uses `ImageDataGenerator` for on-the-fly data augmentation (rotation) on the training set.
4.  **Builds and compiles the model:** As described in the "Model Architecture" section.
5.  **Trains the model:** Trains for 10 epochs with a batch size of 8. The `fit_generator` method is used.
6.  **Evaluates the model:** Predicts on the test set and prints a classification report and a confusion matrix.
7.  **Plots results:** Generates and saves a plot (`plot.jpg`) showing training/validation loss and accuracy over epochs.

Based on the provided notebook outputs, the model achieves an accuracy of approximately 96.15% on the test set.

## Output Files

*   `plot.jpg`: A plot showing the training and validation loss and accuracy curves over epochs.

## Files in the Repository

*   `README.md`: This file.
*   `brain_tumor_detection.ipynb`: Jupyter Notebook containing the complete code for data loading, preprocessing, model building, training, and evaluation.
*   `brain_tumor_detection.py`: Python script version of the Jupyter Notebook.
