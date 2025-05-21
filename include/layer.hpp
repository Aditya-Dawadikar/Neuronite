#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

class Layer{
    public:
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& grad_output) = 0;
        virtual void update(double learning_rate) = 0;
        virtual ~Layer() = default;
};

#endif