#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"

class Layer{
    public:
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& grad_output) = 0;
        virtual void update(double learning_rate) = 0;
        virtual std::string get_name() const = 0;
        virtual std::pair<int,int> get_input_shape() const = 0;
        virtual std::pair<int,int> get_output_shape() const = 0;
        virtual int param_count() const=0;
        virtual ~Layer() = default;
};

#endif