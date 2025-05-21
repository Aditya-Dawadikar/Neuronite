#include <algorithm>
#include "matrix.hpp"
#include "activation_sigmoid.hpp"
#include <cmath>

/// Static sigmoid function
/// σ(x) = 1 / (1 + e^(-x))
double ActivationSigmoid::sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}

/// Forward pass for sigmoid activation
/// For each input x, computes:
///     y = σ(x) = 1 / (1 + e^(-x))
///
/// Also stores the output σ(x) in `output_cache`
/// for use in the backward pass.
Matrix ActivationSigmoid::forward(const Matrix& input){
    Matrix output = Matrix(input.rows, input.cols);

    for(int i=0;i<input.rows;++i){
        for(int j=0;j<input.cols;++j){
            output.data[i][j] = ActivationSigmoid::sigmoid(input.data[i][j]);
        }
    }

    output_cache = output;

    return output;
}

/// Backward pass for sigmoid
/// Given upstream gradient dL/dy, computes:
///     dL/dx = dL/dy * σ(x) * (1 - σ(x))
///
/// This uses the derivative of the sigmoid function:
///     dσ/dx = σ(x) * (1 - σ(x))
Matrix ActivationSigmoid::backward(const Matrix& grad_output){
    Matrix grad_input = Matrix(grad_output.rows, output_cache.cols);

    for(int i=0;i<grad_input.rows;++i){
        for(int j=0;j<grad_input.cols;++j){
            grad_input.data[i][j] = grad_output.data[i][j] * output_cache.data[i][j] * (1.0 - output_cache.data[i][j]);
        }
    }

    return grad_input;
}

/// No-op update — sigmoid has no learnable parameters
void ActivationSigmoid::update(double learning_rate){
    return;
}