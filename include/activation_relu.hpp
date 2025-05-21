#ifndef ACTIVATION_RELU_HPP
#define ACTIVATION_RELU_HPP

#include "layer.hpp"

class ActivationReLU: public Layer{
    private:
        Matrix mask;
    
    public:
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;
        void update(double learning_rate) override;
};

#endif