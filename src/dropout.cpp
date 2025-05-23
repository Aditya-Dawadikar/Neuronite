#include "dropout.hpp"
#include "utils_random.hpp"
#include <random>

Dropout::Dropout(double p) : drop_probability(p), is_training(true) {}

void Dropout::set_training(bool training) {
    is_training = training;
}

Matrix Dropout::forward(const Matrix& input) {
    input_shape = {input.rows, input.cols};
    Matrix output = input;

    if (is_training) {
        mask = Matrix(input.rows, input.cols);
        for (int i = 0; i < input.rows; ++i)
            for (int j = 0; j < input.cols; ++j) {
                mask.data[i][j] = (random_double(0.0, 1.0) > drop_probability) ? 1.0 : 0.0;
                output.data[i][j] *= mask.data[i][j];
            }
    } else {
        // Scale output by (1 - p) at inference
        for (int i = 0; i < input.rows; ++i)
            for (int j = 0; j < input.cols; ++j)
                output.data[i][j] *= (1.0 - drop_probability);
    }

    return output;
}

Matrix Dropout::backward(const Matrix& grad_output) {
    Matrix grad_input = grad_output;
    if (is_training) {
        for (int i = 0; i < grad_output.rows; ++i)
            for (int j = 0; j < grad_output.cols; ++j)
                grad_input.data[i][j] *= mask.data[i][j];  // apply same dropout mask
    }
    return grad_input;
}

std::string Dropout::get_name() const {
    return "Dropout";
}

std::pair<int, int> Dropout::get_input_shape() const {
    return input_shape;
}

std::pair<int, int> Dropout::get_output_shape() const {
    return input_shape;
}
