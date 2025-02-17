//
// Created by jewoo on 2025-02-13.
//
#pragma once

#include "approximator.h"
#include <string>


namespace rlagent {
    class Environment {
    public:
        Environment() {
        }

        virtual int getNumberOfActions() = 0;

        virtual int getStateDim() = 0;

        virtual VectorXD getState() = 0;

        virtual void step(int action,
                          Eigen::Ref<VectorXD> observation,
                          bool &done) = 0;

        virtual void reset(Eigen::Ref<VectorXD> observation) = 0;

        void reset() {
            VectorXD obs = VectorXD::Zero(getStateDim());
            reset(obs);
        }

        virtual void render(std::string mode = "console") = 0;
    };
}
