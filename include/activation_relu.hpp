#ifndef ACTIVATION_RELU_HPP
#define ACTIVATION_RELU_HPP

#include "activation.hpp"

class ActivationReLU: public Activation{
    private:
        Matrix mask;
    
    public:
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;
};

#endif