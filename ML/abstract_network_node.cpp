//
// Created by jewoo on 2025-02-12.
//
#include "abstract_network_node.h"
#include <cassert>

namespace nnetcpp {
    AbstractNetworkNode::~AbstractNetworkNode() {
        for (AbstractNode *node: _nodes) {
            delete node;
        }
    }

    void AbstractNetworkNode::addNode(AbstractNode *node) {
        _nodes.push_back(node);
    }

    void AbstractNetworkNode::serialize(NetworkSerializer &serializer) {
        for (AbstractNode *node: _nodes) {
            node->serialize(serializer);
        }
    }

    void AbstractNetworkNode::deserialize(NetworkSerializer &serializer) {
        for (AbstractNode *node: _nodes) {
            node->deserialize(serializer);
        }
    }

    void AbstractNetworkNode::forward() {
        for (AbstractNode *node: _nodes) {
            node->forward();
        }
    }

    void AbstractNetworkNode::backward() {
        for (int i = _nodes.size() - 1; i >= 0; --i) {
            AbstractNode *node = _nodes[i];
            node->backward();
        }
    }

    void AbstractNetworkNode::update() {
        for (AbstractNode *node: _nodes) {
            node->update();
        }
    }

    void AbstractNetworkNode::clearError() {
        for (AbstractNode *node: _nodes) {
            node->clearError();
        }
    }

    void AbstractNetworkNode::reset() {
        AbstractNode::reset();

        for (AbstractNode *node: _nodes) {
            node->reset();
        }
    }

    void AbstractNetworkNode::setCurrentTimestep(unsigned int timestep) {
        for (AbstractNode *node: _nodes) {
            node->setCurrentTimestep(timestep);
        }
    }
}
