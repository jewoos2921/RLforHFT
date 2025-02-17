//
// Created by jewoo on 2025-02-17.
//
#include "epsilon_greedy.h"

namespace rlagent {
    int EpsilonGreedy::apply(const Eigen::Ref<const VectorXD> &state) {
        if (_distribution_real(_random_generator) < 1.0 - _epsilon) {
            auto q_values = _approximator->predict(state, _actions);
            int index = 0;
            for (int i = 1; i < q_values.size(); i++) {
                if (q_values[i] > q_values[index]) index = i;
            }
            return index;
        }
        return _distribution_int(_random_generator);
    }
}
