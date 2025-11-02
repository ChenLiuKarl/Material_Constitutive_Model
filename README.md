# Material Constitutive Model Training Using Neural Networks

This project contains two neural network models designed to **train material constitutive models** based on measured **strain** and **stress** data.

---

## 🧠 Overview

The goal of this project is to predict material behavior under different conditions. Two neural network architectures are implemented:

1. **Dense Fully-Connected Neural Network (FCNN)**  
2. **Recurrent Neural Network (RNN)**

These models allow flexible training for materials with or without memory effects.

---

## ⚙️ Models

### 1. Dense FCNN
The **Fully-Connected Neural Network** is used for materials where the relationship between strain and stress can be represented as a direct mapping.

**Key features:**
- Input and output size: typically **6 dimensions** for strain and stress
- Adjustable hyperparameters:
  - Number of layers
  - Layer size
  - Activation function (non-linearity)
- Optimized using **SGD**  
- Includes a **learning rate (LR) scheduler**  

---

### 2. Recurrent Neural Network (RNN)
The **RNN** is designed for **materials with memory**, such as viscoelastic materials, where current stress depends on past strain history.

**Key features:**
- Adjustable hyperparameters:
  - Hidden variable size
  - Hidden network architecture
  - Input network architecture
- Optimized using **Adam**  
- Includes a **learning rate (LR) scheduler**  
- Suitable for modeling time-dependent material behavior

---

## 🙏 Acknowledgments

Special thanks to Dr. Burigede Liu and Ms. Rui Wu for providing the initial code framework as part of Course 4C11 at CUED.
