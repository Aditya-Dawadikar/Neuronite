#pragma once
#include "layer.hpp"
#include "matrix.hpp"
#include <vector>

class Dropout : public Layer {
private:
    double drop_probability;
    Matrix mask;
    bool is_training;
    std::pair<int, int> input_shape;

public:
    Dropout(double p);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;
    void update(double learning_rate) override {}

    void set_training(bool training);

    std::string get_name() const override;
    std::pair<int, int> get_input_shape() const override;
    std::pair<int, int> get_output_shape() const override;
    int param_count() const override { return 0; }
};
