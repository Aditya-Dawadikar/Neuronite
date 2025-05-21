#include "model.hpp"
#include "layer.hpp"

/// Adds a layer to the model
/// Layers are stored in a sequential order for forward and backward chaining
void Model::add(Layer* layer){
    layers.push_back(layer);
}

/// Performs the forward pass through all layers
///
/// Each layer applies a transformation:
///     x_{i+1} = layer_i.forward(x_i)
/// Final output is returned (typically used for loss computation)
Matrix Model::forward(const Matrix& input){
    Matrix out = input;
    for(auto& layer : layers){
        out = layer->forward(out);
    }
    return out;
}

/// Performs the backward pass (backpropagation) through all layers in reverse
///
/// Each layer receives the gradient from the layer above:
///     grad_{i} = layer_{i}.backward(grad_{i+1})
///
/// This propagates gradients from loss back to the first layer
Matrix Model::backward(const Matrix& grad_output){
    Matrix grad = grad_output;
    for(auto it = layers.rbegin(); it!=layers.rend();++it){
        grad = (*it)->backward(grad);
    }
    return grad;
}

/// Updates all layers using their stored gradients and a learning rate
///
/// Typically used after `forward` + `backward` to perform one optimization step
void Model::update(double learning_rate){
    for(auto& layer: layers){
        layer->update(learning_rate);
    }
}