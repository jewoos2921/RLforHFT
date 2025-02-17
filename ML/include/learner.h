//
// Created by jewoo on 2025-02-17.
//

#pragma once

#include <omp.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>
#include <time.h>

#include "approximator.h"
#include "policy.h"
#include "environment.h"

namespace rlagent {
    class Learner {
    public:
        typedef std::function<double(VectorXD, int, VectorXD, Environment *)> reward_function;
        typedef std::function<std::shared_ptr<Environment>(void)> environment_function;
        bool _verbose;
        const std::shared_ptr<Approximator> _approximator;
        reward_function _reward;
        environment_function _environment_generator;

        Learner(
            std::shared_ptr<Approximator> approximator,
            reward_function reward,
            environment_function environment_generator): _verbose(false), _approximator(std::move(approximator)),
                                                         _reward(std::move(reward)),
                                                         _environment_generator(std::move(environment_generator)) {
        }

        virtual std::shared_ptr<Policy> get_policy() {
            throw std::logic_error("Not Implemented");
        }

        void learn(
            int episodes, int max_steps_per_episode,
            std::vector<double> &msve_per_episode_out,
            std::vector<double> &total_reward_per_episode_out) {
            msve_per_episode_out.resize(episodes);
            total_reward_per_episode_out.resize(episodes);
            time_t last_msg = time(0) - 5;

#pragma omp parallel for ordered schedule(dynamic, 1)
            for (int episode = 0; episode < episodes; episode++) {
                double ssve_buffer = 0.0;
                double total_reward_buffer = 0.0;
                std::shared_ptr<Environment> environment = _environment_generator();
                environment.reset();
                learn_episode(max_steps_per_episode,
                              &ssve_buffer, &total_reward_buffer, environment.get());
#pragma omp ordered
                {
                    double msve = ssve_buffer / max_steps_per_episode;
                    msve_per_episode_out[episode] = msve;
                    total_reward_per_episode_out[episode] = total_reward_buffer;
                    if (_verbose && time(0) - last_msg > 5) {
                        last_msg += 5;
                        std::cout << ".--------------------------------------------." << std::endl;
                        std::cout << "| Episode " << episode << " of " << episodes << std::endl;
                        std::cout << "| SVE Mean: " << msve << std::endl;
                        std::cout << "| Total Reward: " << total_reward_buffer << std::endl;
                        std::cout << "'--------------------------------------------'" << std::endl;
                        std::cout << std::endl << std::flush;
                    }
                }
            }
        }

    protected:
        virtual void learn_episode(int max_steps,
                                   double *ssve_out,
                                   double *total_reward_out,
                                   Environment *environment) = 0;
    };
}
