# Digit Classification Using Multi-Layer Perceptron (MLP)

This project implements a Multi-Layer Perceptron (MLP) to classify handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras, and achieves a test accuracy of **97.52%**. The repository includes all steps necessary to load the dataset, preprocess the data, build the model, and evaluate its performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview
This project builds an MLP for the classification of handwritten digits. It includes:

- Loading the MNIST dataset.
- Preprocessing the data by normalizing pixel values and reshaping input data.
- Constructing a feedforward neural network using ReLU and softmax activation functions.
- Training the model for 10 epochs.
- Evaluating the model's accuracy and visualizing the loss and accuracy during training.

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a well-known benchmark dataset consisting of 60,000 training images and 10,000 test images of handwritten digits (0-9), where each image is a grayscale 28x28 pixel representation of a digit.

- **Input Size**: 28x28 pixels (flattened into a vector of 784 features)
- **Classes**: 10 (digits 0-9)

## Model Architecture
The architecture of the MLP used in this project includes:

1. **Input Layer**: 
   - Flattened input of shape (28 * 28 = 784)
   
2. **Hidden Layers**:
   - Dense layer with 128 units and ReLU activation
   - Dense layer with 64 units and ReLU activation

3. **Output Layer**:
   - Dense layer with 10 units (one for each digit) and softmax activation for multi-class classification

4. **Loss Function**: 
   - Sparse Categorical Crossentropy

5. **Optimizer**: 
   - Adam

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ahmdmohamedd/digit-classification-MLP.git
   cd digit-classification-MLP
   ```

2. **Create a Conda Environment**
   ```bash
   conda create --name mlp-digits python=3.8
   conda activate mlp-digits
   ```

3. **Install Required Packages**
   ```bash
   pip install tensorflow numpy matplotlib
   ```

4. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

## Usage

Open `MLP_digit_classification.ipynb` in Jupyter Notebook and run the cells sequentially to:

1. Load the MNIST dataset.
2. Preprocess the images.
3. Define and compile the MLP model.
4. Train the model and evaluate its performance.
5. Visualize the results of the model, including accuracy and loss curves.
6. View predictions made by the model on the test dataset.

## Results

- The model achieves an accuracy of **97.52%** on the test dataset.
- Training and validation accuracy/loss graphs are plotted to show the model's performance over 10 epochs.
- Sample predictions are visualized, comparing predicted labels with true labels.

## Visualizations

Here are some visualizations included in the notebook:

- **Training and Validation Accuracy**:
  A graph showing how the accuracy improves over epochs.
  
- **Training and Validation Loss**:
  A graph showing the reduction in loss over epochs.

- **Sample Predictions**:
  A display of actual digit images alongside the model’s predicted labels.

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
