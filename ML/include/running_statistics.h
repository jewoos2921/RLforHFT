//
// Created by jewoo on 2025-02-16.
//

#pragma once
#include <eigen3/Eigen/Dense>
#include "json.hpp"
#include "serializable.h"

namespace ppo_cpp {
    typedef Eigen::Matrix<float, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor> Mat;

    class RunningStatistics: public virtual ISerializable {

    };
}
