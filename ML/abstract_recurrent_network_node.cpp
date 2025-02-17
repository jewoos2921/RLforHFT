//
// Created by jewoo on 2025-02-12.
//
#include "abstract_recurrent_network_node.h"
#include <cassert>

namespace nnetcpp {
    AbstractRecurrentNetworkNode::BpttVariant AbstractRecurrentNetworkNode::bpttVariant =
            AbstractRecurrentNetworkNode::Standard;

    AbstractRecurrentNetworkNode::AbstractRecurrentNetworkNode() : _timestep(0) {
    }

    AbstractRecurrentNetworkNode::~AbstractRecurrentNetworkNode() {
        reset();
    }

    void AbstractRecurrentNetworkNode::addRecurrentNode(AbstractNode *node) {
        N n;
        n.node = node;
        _recurrent_nodes.push_back(n);
    }

    void AbstractRecurrentNetworkNode::forward() {
        AbstractNetworkNode::forward();
        forwardRecurrent();
    }

    void AbstractRecurrentNetworkNode::backward() {
        AbstractNetworkNode::backward();
        backwardRecurrent();
    }

    void AbstractRecurrentNetworkNode::forwardRecurrent() {
        for (N &n: _recurrent_nodes) {
            assert(n.storage .size() > _timestep);

            n.storage[_timestep]->value = n.node->output()->value;
        }
    }

    void AbstractRecurrentNetworkNode::backwardRecurrent() {
        if (_timestep > 0) {
            for (N &n: _recurrent_nodes) {
                assert(n.storage.size() > _timestep);

                switch (bpttVariant) {
                    case Standard:
                        n.storage[_timestep - 1]->error = (
                            n.node->output()->error - n.storage[_timestep]->error
                        ).cwiseMin(10.0f).cwiseMax(-10.0f);
                        break;
                    case Experimental:
                        n.storage[_timestep - 1]->error = n.node->output()->error * _error_normalization;
                        break;
                }
            }
        }
    }

    void AbstractRecurrentNetworkNode::reset() {
        AbstractNetworkNode::reset();

        for (N &n: _recurrent_nodes) {
            for (Port *port: n.storage) {
                delete port;
            }
            n.storage.clear();
        }

        _max_timestep = 0;
    }

    void AbstractRecurrentNetworkNode::setCurrentTimestep(unsigned int timestep) {
        AbstractNetworkNode::setCurrentTimestep(timestep);

        for (N &n: _recurrent_nodes) {
            assert(timestep <= n.storage.size());

            if (timestep == n.storage.size()) {
                int size = n.node->output()->value.rows();

                n.storage.push_back(new Port);

                n.storage.back()->value = Vector::Zero(size);
                n.storage.back()->error = Vector::Zero(size);
            }

            if (timestep > 0) {
                n.node->output()->value = n.storage[timestep - 1]->value;
            } else {
                n.node->output()->value.setZero();
            }

            n.node->output()->error = n.storage[timestep]->error;
        }

        _timestep = timestep;

        _max_timestep = std::max(_max_timestep, timestep);
        _error_normalization = 1.0f / float(_max_timestep);
    }
}
