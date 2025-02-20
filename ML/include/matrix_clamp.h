//
// Created by jewoo on 2025-02-19.
//

#pragma once
#include <eigen3/Eigen/Dense>

namespace ppo_cpp {
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


    class MatrixClamp {
    public:
        MatrixClamp(const Mat &like, float clamp): MatrixClamp(like.rows(), like.cols(), -clamp, clamp) {
        }

        MatrixClamp(const Mat &like, float low, float high): MatrixClamp(like.rows(), like.cols(), low, high) {
        }

        MatrixClamp(int rows, int cols, float clamp): MatrixClamp(rows, cols, -clamp, clamp) {
        }

        MatrixClamp(int rows, int cols, float low, float high): _lo{Mat::Ones(rows, cols) * low},
                                                                _hi{Mat::Ones(rows, cols) * high} {
        }

        Mat clamp(const Mat &mat) const {
            Mat result = mat.cwiseMax(_lo).cwiseMin(_hi);
            return std::move(result);
        }

    private:
        Mat _lo;
        Mat _hi;
    };
}
