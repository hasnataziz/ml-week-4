# ml-week-4
# Fashion MNIST Image Classification using TensorFlow and Keras

This repository contains a simple neural network implementation for classifying images from the Fashion MNIST dataset using TensorFlow and Keras. The project demonstrates the fundamental steps of building, training, and evaluating a deep learning model.

## Project Overview

The objective of this project is to build a Convolutional Neural Network (CNN) to classify grayscale images of fashion items into 10 different categories. The dataset consists of 70,000 images (60,000 for training and 10,000 for testing), each of size 28x28 pixels.

## Dataset

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a collection of Zalando's article images. Each example is a 28x28 grayscale image, associated with a label from 10 classes:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

The dataset is directly available in TensorFlow/Keras datasets.

## Project Structure

- `fashion_mnist_classification.ipynb`: Jupyter notebook containing the full code for the project.
- `best_model.keras`: The best model saved during training using ModelCheckpoint.
- `README.md`: Project documentation.

## Getting Started

### Prerequisites

To run this project, you need to have the following installed:

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/fashion-mnist-classification.git
    cd fashion-mnist-classification
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn
    ```

3. Open the Jupyter notebook to run the code:
    ```bash
    jupyter notebook fashion_mnist_classification.ipynb
    ```

## Model Architecture

The CNN model consists of the following layers:
- **Conv2D**: 32 filters, kernel size (3, 3), activation='relu'
- **MaxPooling2D**: pool size (2, 2)
- **Flatten**
- **Dense**: 128 units, activation='relu'
- **Dense**: 10 units (output layer), activation='softmax'

The model is compiled using the Adam optimizer and sparse categorical crossentropy as the loss function.

## Training and Evaluation

- The model is trained on the training dataset with 20% of it used for validation.
- Early stopping and model checkpointing are used to prevent overfitting and save the best model.
- The final model is evaluated on the test dataset, achieving a test accuracy of approximately XX%.

## Visualizations

The project includes visualizations for:
- Training and validation accuracy over epochs.
- Training and validation loss over epochs.
- Confusion matrix for model predictions on the test set.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research.
- TensorFlow and Keras for providing easy-to-use deep learning tools.

## Contributing

Feel free to fork this project, submit issues and pull requests, or simply star it if you found it useful!

## Contact

For any questions or inquiries, please contact [Your Name](mailto:youremail@example.com).

