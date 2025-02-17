//
// Created by jewoo on 2025-02-13.
//

#pragma once
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace rlagent {
    typedef Eigen::VectorXf VectorXF;
    typedef Eigen::VectorXd VectorXD;
    typedef Eigen::VectorXi VectorXI;

    class Approximator {
    protected:
        std::vector<omp_lock_t> action_locks;

        virtual double predict_implementation(
            Eigen::Ref<const VectorXF> state, int action) {
            throw std::logic_error("Not Implemented");
        }

        virtual VectorXD predict_implementation(
            Eigen::Ref<const VectorXD> states,
            const Eigen::Ref<const VectorXI> &actions) {
            VectorXD td_error;
            td_error = VectorXD::Zero(actions.size());

            for (int action_idx = 0; action_idx < actions.size(); action_idx++) {
                int action = actions[action_idx];
                omp_set_lock(&action_locks[action]);
                td_error[action_idx] = predict_implementation(
                    states, actions[action_idx]);
                omp_unset_lock(&action_locks[action]);
            }

            return td_error;
        }

        virtual double update_implementation(
            const Eigen::Ref<const VectorXD> state, int action, double target) {
            throw std::logic_error("Not Implemented");
        }

    public:
        const int _number_of_actions;
        const int _dimensions_of_statespace;

        Approximator(int number_of_actions, int dimensions_of_statespace): _number_of_actions(number_of_actions),
                                                                           _dimensions_of_statespace(
                                                                               dimensions_of_statespace) {
            action_locks.resize(number_of_actions);
            for (auto &lck: action_locks) omp_init_lock(&lck);
        }

        virtual ~Approximator() {
            for (auto &lck: action_locks) omp_destroy_lock(&lck);
        }

        virtual void save(std::string filename) = 0;

        virtual void load(std::string filename) = 0;

        virtual VectorXD predict(
            const Eigen::Ref<const VectorXD> &states,
            const Eigen::Ref<const VectorXI> &actions) {
            if (states.size() != _dimensions_of_statespace) {
                throw std::invalid_argument("State vector has wrong size.");
            }
            if (actions.minCoeff() < 0 || actions.maxCoeff() >= _number_of_actions) {
                throw std::invalid_argument("Action vector has wrong size.");
            }

            return predict_implementation(states, actions);
        }

        virtual double update(
            const Eigen::Ref<const VectorXD> &state, int action, double target) {
            if (state.size() != _dimensions_of_statespace) {
                throw std::invalid_argument("State vector has wrong size.");
            }
            if (action < 0 || action >= _number_of_actions) {
                throw std::invalid_argument("Action is out of range.");
            }

            double dt_error;
            omp_set_lock(&action_locks[action]);

            dt_error = update_implementation(state, action, target);
            omp_unset_lock(&action_locks[action]);
            return dt_error;
        }
    };
}
