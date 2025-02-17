//
// Created by jewoo on 2025-02-12.
//

#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

namespace nnetcpp {
    class NetworkSerializer;
    typedef Eigen::VectorXf Vector;
    typedef Eigen::MatrixXf Matrix;
    typedef float Float;

    class AbstractNode {
    public:
        struct Port {
            Vector value;
            Vector error;
        };

        AbstractNode() {
        }

        virtual ~AbstractNode() {
        }

        virtual void serialize(NetworkSerializer &serializer) { (void) serializer; }
        virtual void deserialize(NetworkSerializer &serializer) { (void) serializer; }

        virtual Port *output() = 0;

        virtual void forward() = 0;

        virtual void backward() = 0;

        virtual void update() = 0;

        virtual void clearError() = 0;

        virtual void reset() {
        }

        virtual void setCurrentTimestep(unsigned int timestep) {
            (void) timestep;
            clearError();
        }
    };
}
