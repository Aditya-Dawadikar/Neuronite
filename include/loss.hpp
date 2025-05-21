#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

class Loss{
    public:
        virtual double forward(const Matrix& prediction, const Matrix& target) = 0;
        virtual Matrix backward() = 0;
        virtual ~Loss() = default;
};

#endif