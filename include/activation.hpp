#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "matrix.hpp"

class Activation{
    public:
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& grad_output) = 0;
        virtual ~Activation() = default;
};

#endif