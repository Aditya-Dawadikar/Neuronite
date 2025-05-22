#include <algorithm>
#include "matrix.hpp"
#include "activation_relu.hpp"

/// Forward pass of ReLU activation
/// ReLU(x) = max(0, x)
///
/// For each element x in input:
///   y = max(0, x)
/// Also store a binary mask (1 if x > 0, 0 otherwise) for use in backprop
Matrix ActivationReLU::forward(const Matrix& input){
    Matrix output = Matrix(input.rows, input.cols);
    mask = Matrix(input.rows, input.cols);

    input_shape = {input.rows, input.cols};

    for(int i=0;i<input.rows;++i){
        for(int j=0;j<input.cols;++j){
            output.data[i][j] = std::max(0.0, input.data[i][j]);
            if(input.data[i][j] > 0){
                mask.data[i][j] = 1;
            }else{
                mask.data[i][j] = 0;
            }
        }
    }

    return output;
}

/// Backward pass of ReLU
/// For upstream gradient dL/dy, we compute:
///   dL/dx = dL/dy * dReLU(x)/dx
///
/// Derivative of ReLU:
///   dReLU(x)/dx = 1 if x > 0, else 0
/// So:
///   dL/dx = dL/dy if x > 0, else 0
///
/// Uses the `mask` from forward pass.
Matrix ActivationReLU::backward(const Matrix& grad_output){
    Matrix grad_input = Matrix(grad_output.rows, grad_output.cols);
    for(int i=0;i<grad_output.rows;++i){
        for(int j=0;j<grad_output.cols;++j){
            grad_input.data[i][j] = grad_output.data[i][j] * mask.data[i][j];
        }
    }
    return grad_input;
}

/// No-op for ReLU — it has no learnable parameters
void ActivationReLU::update(double learning_rate){
    return;
}


std::string ActivationReLU:: get_name() const {
    return "ReLU";
}

std::pair<int,int> ActivationReLU::get_input_shape() const {
    return input_shape;
}

std::pair<int,int> ActivationReLU::get_output_shape() const {
    return input_shape;
}

int ActivationReLU::param_count() const{
    return 0;
}