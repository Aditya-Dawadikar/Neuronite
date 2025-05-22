#ifndef ACTIVATION_RELU_HPP
#define ACTIVATION_RELU_HPP

#include "layer.hpp"

class ActivationReLU: public Layer{
    private:
        Matrix mask;

        std::pair<int,int> input_shape;
    
    public:
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;
        void update(double learning_rate) override;
        std::string get_name() const override;
        std::pair<int,int> get_input_shape() const override;
        std::pair<int,int> get_output_shape() const override;
        int param_count() const override;
};

#endif