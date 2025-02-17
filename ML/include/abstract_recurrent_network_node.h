//
// Created by jewoo on 2025-02-12.
//

#pragma once

#include "abstract_network_node.h"
#include <vector>

namespace nnetcpp {
    class AbstractRecurrentNetworkNode : public AbstractNetworkNode {
    public:
        enum BpttVariant {
            Standard,
            Experimental
        };

        static BpttVariant bpttVariant;

        AbstractRecurrentNetworkNode();

        virtual ~AbstractRecurrentNetworkNode();

        void addRecurrentNode(AbstractNode *node);

        void forward() override;

        void backward() override;

        void reset() override;

        void setCurrentTimestep(unsigned int timestep) override;

        unsigned int currentTimestep();

    protected:
        void forwardRecurrent();

        void backwardRecurrent();

    private:
        struct N {
            AbstractNode *node;
            std::vector<Port *> storage;
        };

        unsigned int _timestep;
        unsigned int _max_timestep;
        float _error_normalization;
        std::vector<N> _recurrent_nodes;
    };
}
