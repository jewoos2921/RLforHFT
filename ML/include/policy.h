//
// Created by jewoo on 2025-02-13.
//

#pragma once

#include "approximator.h"
#include <utility>
#include <memory>


namespace rlagent {
    class Policy {
    protected:
        std::shared_ptr<Approximator> _approximator;

    public:
        Policy(std::shared_ptr<Approximator> approximator)
            : _approximator(std::move(approximator)) {
        }

        virtual int apply(
            const Eigen::Ref<const VectorXD> &state) = 0;
    };
}
