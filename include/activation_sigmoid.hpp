#ifndef ACTIVATION_SIGMOID_HPP
#define ACTIVATION_SIGMOID_HPP

#include "layer.hpp"

class ActivationSigmoid: public Layer{
    private:
        Matrix mask;
        Matrix output_cache;
    
    public:
        static double sigmoid(double x);

        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& grad_output) override;
        void update(double learning_rate) override;
};

#endif