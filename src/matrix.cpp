#include <matrix.hpp>
#include <iomanip>

Matrix::Matrix():rows(0),cols(0),data() {}

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

Matrix::Matrix(int rows, int cols, double init_val)
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {
        for (int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                data[i][j] = init_val;
            }
        }
    }

Matrix::Matrix(const std::vector<std::vector<double>>& values)
    : rows(values.size()), cols(values[0].size()), data(values) {}

Matrix Matrix::col_sum() const{
    Matrix result = Matrix(1,cols);

    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            result.data[0][j] += data[i][j];
        }
    }

    return result;
}

Matrix Matrix::dot(const Matrix& A, const Matrix& B){

    if (A.cols != B.rows){
        throw std::invalid_argument("Dot: Incompatible dimensions");
    }

    Matrix result(A.rows, B.cols);
    for(int i=0;i<A.rows;++i){
        for(int j=0;j<B.cols;++j){
            for(int k=0;k<A.cols;++k){
                result.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix&other) const{
    Matrix result(rows, cols);

    if (rows == other.rows && cols == other.cols){
        // matrices with same dimension
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
    }else if (other.rows == 1 && other.cols == cols){
        // 2D matrix + 1D matrix
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                result.data[i][j] = data[i][j] + other.data[0][j];
            }
        }
    }else{
        throw std::invalid_argument("Matrix::operator+: Shape mismatch for addition.");
    }

    return result;
}

Matrix Matrix::operator-(const Matrix&other) const{
    Matrix result(rows, cols);

    if (rows == other.rows && cols == other.cols){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
    }else if(other.rows == 1 && other.cols == cols){
        // 2D matrix - 1D matrix
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                result.data[i][j] = data[i][j] - other.data[0][j];
            }
        }
    }else{
        throw std::invalid_argument("Matrix::operator-: Shape mismatch for subtraction.");
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const{
    Matrix result(rows, cols);

    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            result.data[i][j] = data[i][j] * scalar;
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix&other) const{
    Matrix result(rows, cols);

    if (rows == other.rows && cols == other.cols){
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
    }else if(other.rows == 1 && other.cols == cols){
        // 2D matrix * 1D matrix
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                result.data[i][j] = data[i][j] * other.data[0][j];
            }
        }
    }else{
        throw std::invalid_argument("Matrix::operator*: Shape mismatch for multiplication.");
    }

    return result;
}

void Matrix::print() const {
    std::cout << "[\n";
    for (const auto& row : data) {
        std::cout << "  [ ";
        for (double val : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}