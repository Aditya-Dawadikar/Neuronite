#ifndef UTILS_RANDOM_HPP
#define UTILS_RANDOM_HPP

#pragma once

#include "matrix.hpp"

void set_random_seed(unsigned int seed);
void initialize_random(Matrix& mat, double min=-1.0, double max = 1.0);
double random_double(double min = 0.0, double max = 1.0);

#endif