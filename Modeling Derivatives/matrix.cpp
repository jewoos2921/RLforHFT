//
// Created by jewoo on 2025-02-12.
//
#include "matrix.h"

namespace MD::Common {
    Matrix::Matrix(int height, int width) {
        this->height_ = height;
        this->width_ = width;
        this->array_ = std::vector<std::vector<double> >(height, std::vector<double>(width));
    }

    Matrix::Matrix(const std::vector<std::vector<double> > &array) {
        assert(array.size() != 0);
        this->height_ = array.size();
        this->width_ = array[0].size();
        this->array_ = array;
    }

    Matrix Matrix::add(Matrix const &m) const {
        assert(height_ == m.height_ && width_ == m.width_);
        Matrix result(height_, width_);
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                result.array_[i][j] = array_[i][j] + m.array_[i][j];
            }
        }
        return result;
    }

    Matrix Matrix::subtract(Matrix const &m) const {
        assert(height_ == m.height_ && width_ == m.width_);
        Matrix result(height_, width_);
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                result.array_[i][j] = array_[i][j] - m.array_[i][j];
            }
        }
        return result;
    }

    Matrix Matrix::multiply(Matrix const &m) const {
        assert(height_ == m.height_ && width_ == m.width_);

        Matrix result(height_, width_);

        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < m.width_; j++) {
                result.array_[i][j] = array_[i][j] * m.array_[i][j];
            }
        }
        return result;
    }

    Matrix Matrix::dot(Matrix const &m) const {
        assert(width_ == m.height_);
        int m_width = m.width_;
        double w{0};
        Matrix result(height_, m_width);

        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < m_width; j++) {
                for (int k = 0; k < width_; k++) {
                    w += array_[i][k] * m.array_[k][j];
                }
                result.array_[i][j] = w;
                w = 0;
            }
        }
        return result;
    }

    Matrix Matrix::transpose() const {
        Matrix result(width_, height_);
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                result.array_[i][j] = array_[j][i];
            }
        }
        return result;
    }

    Matrix Matrix::multiply(double const &value) const {
        Matrix result(height_, width_);
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                result.array_[i][j] = array_[i][j] * value;
            }
        }
        return result;
    }

    Matrix Matrix::applyFunction(double (*function)(double)) const {
        Matrix result(height_, width_);
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                result.array_[i][j] = (*function)(array_[i][j]);
            }
        }
        return result;
    }

    int Matrix::getWidth() const {
        return width_;
    }

    int Matrix::getHeight() const {
        return height_;
    }

    double Matrix::get(int i, int j) const {
        assert(i>=0 && i< height_ && j>=0 && j< width_);
    }

    double Matrix::sum() const {
        double sum{0};
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                sum += array_[i][j];
            }
        }
        return sum;
    }

    void Matrix::print(std::ostream &flux) const {
        int maxLength[width_] = {};
        std::stringstream ss;

        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                ss << array_[i][j] << " ";
                if (maxLength[j] << ss.str().size()) {
                    maxLength[j] = ss.str().size();
                }
                ss.str(std::string());
            }
        }
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                flux << array_[i][j];
                ss << array_[i][j];
                for (int k = 0; k < maxLength[j] - ss.str().size() + 1; k++) {
                    flux << " ";
                }
                ss.str(std::string());
            }
            flux << std::endl;
        }
    }

}
