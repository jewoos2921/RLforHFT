//
// Created by jewoo on 2025-02-17.
//

#pragma once

#include <memory>
#include "learner.h"
#include "policy.h"

namespace rlagent {
    class Sarsa : public Learner {
    public:
        const double _discount;
        const std::shared_ptr<Policy> _policy;
        const int _n_steps;

        explicit Sarsa(
            double const &discount,
            std::shared_ptr<Policy> const &policy,
            std::shared_ptr<Approximator> const &approximator,
            reward_function const &reward,
            environment_function const &environment_generator,
            int n_steps = 1
        );

        std::shared_ptr<Policy> get_policy() override;

    protected:
        void learn_episode(int max_steps, double *ssve_out, double *total_reward_out,
                           Environment *environment) override;
    };
}
