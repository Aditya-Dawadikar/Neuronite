#include "adam_optimizer.hpp"
#include "dense_layer.hpp"
#include <cmath>

AdamOptimizer::AdamOptimizer(double lr, double beta1, double beta2, double epsilon)
    : lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

/// Performs one step of the Adam optimization algorithm on a single layer
/// 
/// Uses the layer's stored gradients (d_weights and d_bias) to compute updates.
/// Maintains per-layer first and second moment estimates for weights and bias.
///
/// Parameters:
/// - layer: pointer to the Layer (must be DenseLayer)
/// - t: current timestep (starting from 1), used for bias correction
void AdamOptimizer::step(Layer* layer, int t) {

    // Only apply Adam to Dense layers (skip ReLU/Sigmoid etc.)
    auto* dense = dynamic_cast<DenseLayer*>(layer);
    if (!dense) return;

    Matrix dw = dense->get_d_weights();
    Matrix db = dense->get_d_bias();

    // Initialize moment vectors if this is the first time seeing this layer
    if (m_weights.count(layer) == 0) {
        m_weights[layer] = Matrix(dw.rows, dw.cols);
        v_weights[layer] = Matrix(dw.rows, dw.cols);
        m_bias[layer] = Matrix(db.rows, db.cols);
        v_bias[layer] = Matrix(db.rows, db.cols);
    }

    // Aliases for convenience
    auto& mw = m_weights[layer]; // First moment (mean) for weights
    auto& vw = v_weights[layer]; // Second moment (variance) for weights
    auto& mb = m_bias[layer];    // First moment (mean) for bias
    auto& vb = v_bias[layer];    // Second moment (variance) for bias

    // Get current weights and biases to update them
    Matrix updated_weights = dense->get_weights();
    Matrix updated_bias = dense->get_bias();

    // ==== Update Weights ====
    for (int i = 0; i < dw.rows; ++i) {
        for (int j = 0; j < dw.cols; ++j) {
            // m ← β1·m + (1−β1)·∇θ
            mw.data[i][j] = beta1 * mw.data[i][j] + (1 - beta1) * dw.data[i][j];

            // v ← β2·v + (1−β2)·(∇θ)^2
            vw.data[i][j] = beta2 * vw.data[i][j] + (1 - beta2) * std::pow(dw.data[i][j], 2);

            // Bias-corrected moment estimates
            double m_hat = mw.data[i][j] / (1 - std::pow(beta1, t));
            double v_hat = vw.data[i][j] / (1 - std::pow(beta2, t));

            // θ ← θ − α·m̂ / (√v̂ + ε)
            updated_weights.data[i][j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    for (int j = 0; j < db.cols; ++j) {
        mb.data[0][j] = beta1 * mb.data[0][j] + (1 - beta1) * db.data[0][j];
        vb.data[0][j] = beta2 * vb.data[0][j] + (1 - beta2) * std::pow(db.data[0][j], 2);

        double m_hat = mb.data[0][j] / (1 - std::pow(beta1, t));
        double v_hat = vb.data[0][j] / (1 - std::pow(beta2, t));

        updated_bias.data[0][j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    // Apply updated parameters to the layer
    dense->apply_adam_update(updated_weights, updated_bias);
}

