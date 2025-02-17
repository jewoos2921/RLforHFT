//
// Created by jewoo on 2025-02-17.
//

#pragma once

#include "policy.h"
#include <random>

namespace rlagent {
    class EpsilonGreedy : public Policy {
    private:
        std::mt19937 _random_generator;
        std::uniform_real_distribution<> _distribution_real;
        std::uniform_int_distribution<> _distribution_int;
        VectorXI _actions;

    public:
        double _epsilon;

        EpsilonGreedy(double epsilon,
                      const std::shared_ptr<Approximator> &approximator): Policy(approximator),
                                                                          _random_generator(std::random_device{}()),
                                                                          _distribution_real(0.0, 1.0),
                                                                          _distribution_int(
                                                                              0, approximator->_number_of_actions - 1),
                                                                          _epsilon(epsilon) {
            _actions = VectorXI ::LinSpaced(approximator->_number_of_actions, 0,
                                             approximator->_number_of_actions - 1);
        }

        int apply(const Eigen::Ref<const VectorXD> &state) override;
    };
}
