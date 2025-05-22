#include "model.hpp"
#include "layer.hpp"
#include <iomanip>
#include <iostream>

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

void Model::summarize(int input_dim){
    // Step 1: Run dummy forward
    Matrix dummy_input(1, input_dim);  // [1 × input_dim], batch size = 1
    for (auto& row : dummy_input.data)
        std::fill(row.begin(), row.end(), 0.0);  // optional: fill with 0s

    this->forward(dummy_input);  // populates input/output shapes in layers

    // Step 2: Print header
    std::cout << "# Model Summary\n";
    std::cout << "──────────────────────────────────────────────────────────────\n";
    std::cout << std::left
              << std::setw(20) << "Layer (type)"
              << std::setw(18) << "Input Shape"
              << std::setw(18) << "Output Shape"
              << std::setw(10) << "Param #" << "\n";
    std::cout << "==============================================================\n";

    int total_params = 0;

    for (const auto& layer : layers) {
        auto in_shape = layer->get_input_shape();
        auto out_shape = layer->get_output_shape();
        int params = layer->param_count();

        std::string in_shape_str = "[" + std::to_string(in_shape.first) + " × " + std::to_string(in_shape.second) + "]";
        std::string out_shape_str = "[" + std::to_string(out_shape.first) + " × " + std::to_string(out_shape.second) + "]";

        std::cout << std::left
                  << std::setw(20) << layer->get_name()
                  << std::setw(18) << in_shape_str
                  << std::setw(18) << out_shape_str
                  << std::setw(10) << params << "\n";

        total_params += params;
    }

    std::cout << "──────────────────────────────────────────────────────────────\n";
    std::cout << "Total Parameters: " << total_params << "\n";
    std::cout << "\n";
}