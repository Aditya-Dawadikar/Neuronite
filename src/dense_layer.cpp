#include "matrix.hpp"
#include "dense_layer.hpp"
#include "utils_random.hpp"
#include <cmath>

// DenseLayer constructor
// Initializes weights and biases, and allocates memory for gradients
// weight shape: (input_dim × output_dim)
// bias shape:   (1 × output_dim)
// d_weights:    (input_dim × output_dim)
// d_bias:       (1 × output_dim)
DenseLayer:: DenseLayer(int input_dim, int output_dim)
    : weights(input_dim, output_dim),
        bias(1, output_dim),
        d_weights(input_dim, output_dim),
        d_bias(1, output_dim){
    
    double limit = std::sqrt(6.0 / (input_dim + output_dim));
    initialize_random(weights, -1*limit, 1*limit);
}

// Forward pass of the dense layer
// input shape:  (batch_size × input_dim)
// output shape: (batch_size × output_dim)
// Computes: Z = X · W + b
Matrix DenseLayer:: forward(const Matrix& input){
    // Cache input for use in backward pass
    input_cache = input;
    input_shape = {input.rows, input.cols};

    // Matrix multiplication: (batch_size × input_dim) · (input_dim × output_dim)
    Matrix output = Matrix::dot(input, weights);

    // Broadcast and add bias: bias is (1 × output_dim)
    output = output + bias;

    output_shape = {output.rows, output.cols};

    return output;
}

// Backward pass of the dense layer
// grad_output = ∂L/∂Z (gradient of loss w.r.t. layer output)
// grad_output shape: (batch_size × output_dim)
// Returns: ∂L/∂X = grad_input, shape: (batch_size × input_dim)
//
// Computes:
// d_weights = Xᵗ · ∂L/∂Z       (input_dim × output_dim)
// d_bias    = sum_rows(∂L/∂Z)  (1 × output_dim)
// grad_input = ∂L/∂Z · Wᵗ      (batch_size × input_dim)
Matrix DenseLayer:: backward(const Matrix& grad_output){
    // ∂L/∂W = inputᵗ · grad_output
    d_weights = Matrix::dot(input_cache.transpose(),grad_output);

    // ∂L/∂b = row-wise sum of grad_output
    d_bias = grad_output.col_sum();

    // ∂L/∂X = grad_output · weightsᵗ
    Matrix grad_input = Matrix::dot(grad_output, weights.transpose());

    return grad_input;
}

// Update step: performs SGD on weights and bias
// W := W - η ∂L/∂W
// b := b - η ∂L/∂b
void DenseLayer:: update(double learning_rate){
    // apply gradient updates
    Matrix scaled_weights = d_weights*learning_rate;
    Matrix scaled_bias = d_bias*learning_rate;

    weights = weights - scaled_weights;
    bias = bias - scaled_bias;
}

Matrix DenseLayer:: get_weights() const{
    return weights;
}

Matrix DenseLayer:: get_bias() const{
    return bias;
}

Matrix DenseLayer:: get_d_weights() const{
    return d_weights;
}

Matrix DenseLayer:: get_d_bias() const{
    return d_bias;
}

std::string DenseLayer:: get_name() const {
    return "Dense("+std::to_string(weights.rows)+" -> "+std::to_string(weights.cols)+")";
}

std::pair<int,int> DenseLayer::get_input_shape() const {
    return input_shape;
}

std::pair<int,int> DenseLayer::get_output_shape() const {
    return output_shape;
}

int DenseLayer::param_count() const{
    int input_dim = input_shape.second;
    int output_dim = output_shape.second;
    return input_dim*output_dim + output_dim;
}

void DenseLayer::apply_adam_update(const Matrix& new_weights, const Matrix& new_bias) {
    weights = new_weights;
    bias = new_bias;
}