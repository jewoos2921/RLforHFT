//
// Created by jewoo on 2025-02-12.
//

#pragma once
#include "abstract_node.h"


namespace nnetcpp {
    class AbstractMergeNode : public AbstractNode {
    public:
        AbstractMergeNode();

        void addInput(Port *input);

        Port *output() override;

        void update() override;

        void clearError() override;

    protected:
        Port _output;
        std::vector<Port *> _inputs;
    };
}
