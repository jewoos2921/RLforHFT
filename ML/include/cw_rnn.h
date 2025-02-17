//
// Created by jewoo on 2025-02-15.
//

#pragma once

#include "abstract_recurrent_network_node.h"
#include "activation.h"
#include <vector>

namespace nnetcpp {
    class Dense;
    class MergeSum;

    class CWRNN : public AbstractRecurrentNetworkNode {
    public:
        CWRNN(unsigned int num_units,
              unsigned int size,
              Float learning_rate, Float decay = 0.9f);

        void addInput(Port *input);

        Port *output() override;

        void forward() override;

        void backward() override;

    private:
        template<class EnableFunc, class DisabledFunc>
        void forUnits(unsigned int t,
                      EnableFunc enabled, DisabledFunc disabled);

        struct Unit {
            std::vector<Dense *> inputs;
            MergeSum *sum;
            TanhActivation *activation;
            LinearActivation *skip;
            MergeSum *output;
        };

        std::vector<Unit> _units;
        std::vector<Dense *> _inputs;
        MergeSum *_output;

        unsigned int _unit_size;
        float _learning_rate;
        float _decay;
    };

    template<class EnableFunc, class DisabledFunc>
    void CWRNN::forUnits(unsigned int t, EnableFunc enabled, DisabledFunc disabled) {
        for (unsigned int i = 0; i < _units.size(); ++i) {
            unsigned int period = 1 << (_units.size() - i - 1);

            if (t % period == 0)
                enabled(_units[i]);
            else
                disabled(_units[i]);
        }
    }
}
