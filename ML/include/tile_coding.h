//
// Created by jewoo on 2025-02-15.
//

#pragma once

#include "state_aggregation.h"

namespace rlagent {
    class TileCoding : public Approximator {
    public:
        double _step_size;

        TileCoding(
            int number_of_actions,
            int dimensions_of_statespace,
            double step_size,
            int tilings,
            const Eigen::Ref<const VectorXI> &displacement,
            const Eigen::Ref<const VectorXI> &segments,
            const Eigen::Ref<const VectorXF> &min_values,
            const Eigen::Ref<const VectorXF> &max_values,
            const Eigen::Ref<const VectorXF> &action_kernel = (Eigen::Matrix<float, 1, 1>() << 1, 0).finished(),
            double init_min_value = 0.0, double init_max_value = 0.0
        );

        void save(std::string filename) override;

        void load(std::string filename) override;

        VectorXD predict(const Eigen::Ref<const VectorXD> &states,
            const Eigen::Ref<const VectorXI> &actions) override;

        double update(const Eigen::Ref<const VectorXD> &state, int action, double target) override;

    private:
        std::vector<StateAggregation> _layers;
    };
}
