//
// Created by jewoo on 2025-02-12.
//

#include <iostream>
#include <vector>
#include "abstract_node.h"

namespace ML {
    class RLEnvironment {
    public:
        int reward(const std::pair<int, int> &state) const {
            if (state == std::make_pair(9, 9)) {
                return 10;
            }
            return -1;
        }

        bool isTerminal(const std::pair<int, int> &state) const {
            return state == std::make_pair(9, 9);
        }
    };

    class RLAgent {
        std::pair<int, int> position;

    public:
        RLAgent() : position(0, 0) {
        }

        void moveTo(int x, int y) {
            position = std::make_pair(x, y);
        }

        std::pair<int, int> getPosition() const {
            return position;
        }
    };
}
