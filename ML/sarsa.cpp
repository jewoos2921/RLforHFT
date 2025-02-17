//
// Created by jewoo on 2025-02-17.
//

#include "sarsa.h"
#include <deque>
#include <utility>
#include <algorithm>
#include <vector>

namespace rlagent {
    Sarsa::Sarsa(double const &discount, std::shared_ptr<Policy> const &policy,
                 std::shared_ptr<Approximator> const &approximator, reward_function const &reward,
                 environment_function const &environment_generator, int n_steps): Learner(approximator, reward,
            environment_generator), _discount(discount), _policy(policy), _n_steps(n_steps) {
    }

    std::shared_ptr<Policy> Sarsa::get_policy() {
        return _policy;
    }

    void Sarsa::learn_episode(int max_steps, double *ssve_out,
                              double *total_reward_out, Environment *environment) {
        Eigen::MatrixXd n_step_states = Eigen::MatrixXd::Zero(environment->getStateDim(), _n_steps);

        std::vector<double> n_step_rewards(_n_steps);
        std::vector<int> n_step_actions(_n_steps);

        VectorXD state = VectorXD::Zero(environment->getStateDim());
        environment->reset(state);


        int action = _policy->apply(state);
        int step = 0;
        int tau = 0;
        *ssve_out = 0;
        *total_reward_out = 0;

        int remaining_steps = max_steps;

        n_step_states.col(step % _n_steps) = state;
        n_step_actions[step % _n_steps] = action;

        while (remaining_steps > 0) {
            do {
                if (step < max_steps) {
                    bool terminal = false;
                    VectorXD next_state = VectorXD::Zero(state.size());

                    environment->step(action, next_state, terminal);
                    double reward_value = _reward(state, action, next_state, environment);
                    *total_reward_out += reward_value;

                    n_step_rewards[(step + 1) % _n_steps] = reward_value;
                    n_step_states.col((step + 1) % _n_steps) = next_state;

                    state = next_state;
                    action = _policy->apply(state);

                    if (terminal) {
                        environment->reset(state);
                        action = _policy->apply(state);
                        max_steps = step + 1;
                    } else {
                        n_step_actions[(step + 1) % _n_steps] = action;
                    }
                }

                tau = step - _n_steps + 1;
                if (tau >= 0) {
                    double reward_sum = 0.0;

                    for (int i = tau + 1; i <= std::min(max_steps, step + 1); i++) {
                        double future_reward = n_step_rewards[i % _n_steps];
                        double dampening = std::pow(_discount, i - tau - 1);
                        reward_sum = reward_sum + dampening * future_reward;
                    }

                    int future_time = tau + _n_steps;
                    if (future_time < max_steps) {
                        auto future_action = Eigen::Map<Eigen::Matrix<int, 1, 1> >(
                            &n_step_actions[future_time % _n_steps]);
                        auto future_state = n_step_states.col(future_time % _n_steps);
                        reward_sum = reward_sum + std::pow(_discount, _n_steps) * _approximator->predict(
                                         future_state, future_action)[0];
                    }
                    double td_error = _approximator->update(
                        n_step_states.col(tau % _n_steps),
                        n_step_actions[tau % _n_steps],
                        reward_sum
                    );
                    *ssve_out = *ssve_out + std::pow(td_error, 2.0);
                }

                step = step + 1;
            } while (tau != max_steps - 1);

            remaining_steps -= max_steps;

            max_steps = remaining_steps;
            step = 0;
        }
    }
}
