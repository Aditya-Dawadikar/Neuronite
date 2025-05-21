#include <algorithm>
#include "matrix.hpp"
#include "activation_relu.hpp"

Matrix ActivationReLU::forward(const Matrix& input){
    Matrix output = Matrix(input.rows, input.cols);
    mask = Matrix(input.rows, input.cols);

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

    std::cout << "Mask:\n";
    mask.print();

    return output;
}

Matrix ActivationReLU::backward(const Matrix& grad_output){
    Matrix grad_input = Matrix(grad_output.rows, grad_output.cols);
    for(int i=0;i<grad_output.rows;++i){
        for(int j=0;j<grad_output.cols;++j){
            grad_input.data[i][j] = grad_output.data[i][j] * mask.data[i][j];
        }
    }
    return grad_input;
}