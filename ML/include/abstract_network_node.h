//
// Created by jewoo on 2025-02-12.
//

#pragma once
#include "abstract_node.h"

namespace nnetcpp {
    class AbstractNetworkNode : public AbstractNode {
    public:
        virtual ~AbstractNetworkNode();

        void addNode(AbstractNode *node);

        void serialize(NetworkSerializer &serializer) override;

        void deserialize(NetworkSerializer &serializer) override;

        void forward() override;

        void backward() override;

        void update() override;

        void clearError() override;;

        void reset() override;

        void setCurrentTimestep(unsigned int timestep) override;

    protected:
        std::vector<AbstractNode *> _nodes;
    };
}
