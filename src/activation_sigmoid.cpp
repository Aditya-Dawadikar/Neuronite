#include <algorithm>
#include "matrix.hpp"
#include "activation_sigmoid.hpp"
#include <cmath>

double ActivationSigmoid::sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}

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

Matrix ActivationSigmoid::backward(const Matrix& grad_output){
    Matrix grad_input = Matrix(grad_output.rows, output_cache.cols);

    for(int i=0;i<grad_input.rows;++i){
        for(int j=0;j<grad_input.cols;++j){
            grad_input.data[i][j] = grad_output.data[i][j] * output_cache.data[i][j] * (1.0 - output_cache.data[i][j]);
        }
    }

    return grad_input;
}