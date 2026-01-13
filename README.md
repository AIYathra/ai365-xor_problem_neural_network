# 🧠 XOR Neural Network — Solving the Classic Non‑Linear Problem

A minimal, hands‑on implementation of the XOR problem using a simple neural network built with TensorFlow/Keras. This project demonstrates how non‑linear activation functions and a hidden layer enable a model to learn a function that a single‑layer perceptron cannot solve.


## 🚀 Project Overview

The XOR (exclusive OR) problem is historically important in the evolution of neural networks. It represents the simplest example of a **non‑linearly separable** function — meaning it cannot be solved by a linear classifier.

This repository walks through:

- Building the XOR dataset  
- Designing a small neural network with a hidden layer  
- Training the model to learn XOR  
- Testing predictions  
- Converting the trained model into TensorFlow Lite format  

This project is intentionally simple and ideal for beginners who want to understand *why* neural networks need non‑linearity and hidden layers.


## 📂 Repository Structure

ai365-xor_problem_neural_network/
```text
├── Xor.ipynb              # Jupyter Notebook with full implementation
├── converted_model.tflite # Exported TensorFlow Lite model
└── README.md              # Project documentation
```


## 🧩 The XOR Dataset

The XOR truth table:

| Input (x1, x2) | Output |
|----------------|--------|
| (0, 0)         |   0    |
| (0, 1)         |   1    |
| (1, 0)         |   1    |
| (1, 1)         |   0    |

This dataset cannot be separated by a straight line, which is why a hidden layer is required.


## 🏗️ Model Architecture

The neural network used in this project:

- **Input layer:** 2 features  
- **Hidden layer:** 2 neurons, `tanh` activation  
- **Output layer:** 1 neuron, `sigmoid` activation  
- **Loss:** Binary cross‑entropy  
- **Optimizer:** Adam  

This minimal architecture is sufficient to learn the XOR mapping.


## 🏃 Training

The model is trained for 10,000 epochs with batch size 1.  
After training, the network correctly predicts the XOR outputs.

Example output:
[[0.01]
[0.98]
[0.97]
[0.02]]


(Rounded → `[[0], [1], [1], [0]]`)


## 📦 TensorFlow Lite Conversion

The trained Keras model is converted into a `.tflite` file using:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```
This enables deployment on microcontrollers or edge devices.


## 🎯 Learning Outcome
By completing this project, learn:
- Why XOR is a foundational problem in neural network history
- How hidden layers and non‑linear activations enable complex decision boundaries
- How to build, train, and test a neural network in TensorFlow/Keras
- How to export a model to TensorFlow Lite
This is a perfect stepping stone toward deeper neural network concepts and embedded AI.


## 📚 Learning Resources
New to neural networks? Check out our detailed tutorial!
📚 **[Line-by-Line Tutorial](xor_tutorial_line_by_line.md)** - Perfect for beginners! Every line of code explained in simple terms.


## 📘 Educational Purpose
This repository is created purely for educational and learning purposes. It is designed to help beginners understand the fundamentals of neural networks through the classic XOR problem.