#include "model.hpp"
#include "layer.hpp"

void Model::add(Layer* layer){
    layers.push_back(layer);
}

Matrix Model::forward(const Matrix& input){
    Matrix out = input;
    for(auto& layer : layers){
        out = layer->forward(out);
    }
    return out;
}

Matrix Model::backward(const Matrix& grad_output){
    Matrix grad = grad_output;
    for(auto it = layers.rbegin(); it!=layers.rend();++it){
        grad = (*it)->backward(grad);
    }
    return grad;
}

void Model::update(double learning_rate){
    for(auto& layer: layers){
        layer->update(learning_rate);
    }
}