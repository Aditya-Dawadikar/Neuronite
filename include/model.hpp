#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include "matrix.hpp"
#include "layer.hpp"

class Model{
    private:
        std::vector<Layer*> layers;
    
    public:
        void add(Layer* layer);
        Matrix forward(const Matrix& loss_grad);
        Matrix backward(const Matrix& loss_grad);
        void update(double learning_rate);
};

#endif