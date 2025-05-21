#ifndef LOSS_MSE_HPP
#define LOSS_MSE_HPP

#include "loss.hpp"

class LossMSE: public Loss{
    private:
        Matrix prediction_cache;
        Matrix target_cache;
    
    public:
        double forward(const Matrix& prediction, const Matrix& target) override;
        Matrix backward() override;
};

#endif