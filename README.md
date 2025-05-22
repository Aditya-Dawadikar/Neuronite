# 🧠 Neuronite

**Neuronite** is a lightweight neural network library written from scratch in C++ — built for learning, experimentation, and full transparency into how neural networks work at the bare metal level.

> ⚠️ This project is a work in progress (WIP). Major components are being added and refined daily. Use at your own curiosity.

---

## 🚀 Features

- Dense (Fully Connected) Layers
- Activation functions: ReLU, Sigmoid
- Loss function: Mean Squared Error (MSE)
- Optimizers: SGD, Adam
- Model summary with input/output dimensions
- Forward and backward propagation
- Early stopping and accuracy tracking
- Modular Layer/Model architecture

---

## 🛠️ Installation

### Prerequisites
- C++17 compiler (e.g. `g++`, `clang++`)
- CMake ≥ 3.10
- Make

### Build

```bash
git clone https://github.com/yourusername/neuronite.git
cd neuronite
mkdir build && cd build
cmake ..
make
```

---

## 🧪 Usage

### 1. Include Required Headers

```cpp
#include "model.hpp"
#include "dense_layer.hpp"
#include "activation_relu.hpp"
#include "activation_sigmoid.hpp"
#include "loss_mse.hpp"
#include "optimizer_adam.hpp"
```

### 2. Build Model

```cpp
Model model;
model.add(new DenseLayer(2, 4));
model.add(new ActivationReLU());
model.add(new DenseLayer(4, 1));
model.add(new ActivationSigmoid());
```

### 3. Define Data and Train

```cpp
Matrix X({
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
});

Matrix y({
    {0},
    {1},
    {1},
    {0}
});

LossMSE loss;
AdamOptimizer optimizer(0.01);  // learning rate

model.summarize(2);
model.train(X, y, loss, optimizer, 500, 30);  // 500 epochs, 30-patience early stop

```

### Sample Output

```yaml
# Model Summary
──────────────────────────────────────────────────────────────
Layer (type)        Input Shape       Output Shape      Param #
==============================================================
Dense(2 -> 4)       [1 × 2]           [1 × 4]           12
ReLU                [1 × 4]           [1 × 4]           0
Dense(4 -> 1)       [1 × 4]           [1 × 1]           5
Sigmoid             [1 × 1]           [1 × 1]           0
──────────────────────────────────────────────────────────────
Total Parameters: 17

Epoch 0   | Loss: 0.26  | Accuracy: 50.00%
Epoch 200 | Loss: 0.01  | Accuracy: 100.00%

```
