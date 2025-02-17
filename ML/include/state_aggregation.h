//
// Created by jewoo on 2025-02-13.
//

#pragma once

#include "approximator.h"


namespace rlagent {
    class StateAggregation : public Approximator {
    public:
        double _step_size;

        StateAggregation(int number_of_actions,
                         int dimensions_of_statespace,
                         double step_size,
                         const Eigen::Ref<const VectorXI> &segments,
                         const Eigen::Ref<const VectorXF> &min_values,
                         const Eigen::Ref<const VectorXF> &max_values,
                         const Eigen::Ref<const VectorXF> &action_kernel = (Eigen::Matrix<float, 1, 1>() << 1.0).
                                 finished(),
                         double init_min_value = 0.0,
                         double init_max_value = 0.0);

        void save(std::string filename) override;

        void load(std::string filename) override;

        Eigen::Ref<VectorXF> getValues();

    private:
        VectorXF _action_kernel;
        VectorXF _values;
        VectorXI _segments;
        VectorXF _segment_size;
        VectorXF _min_values;
        VectorXF _max_values;

        VectorXI get_indices(VectorXD state);

    protected:
        double predict_implementation(Eigen::Ref<const VectorXF> state, int action) override;

        double update_implementation(const Eigen::Ref<const VectorXD> state, int action, double target) override;
    };
}
