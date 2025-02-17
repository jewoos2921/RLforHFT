//
// Created by jewoo on 2025-02-13.
//

#include "network_serializer.h"
#include <cassert>

namespace nnetcpp {
    NetworkSerializer::NetworkSerializer(): _pos(0) {
    }

    void NetworkSerializer::writeWeight(float value) {
        _data.push_back(value);
    }

    float NetworkSerializer::readWeight() {
        assert(_pos < _data.size());
        return _data[_pos++];
    }

    void NetworkSerializer::save(std::ostream &s) {
        s.write((const char *) _data.data(),
                _data.size() * sizeof(float));
    }

    void NetworkSerializer::load(std::istream &s) {
        float v;

        while (!s.eof()) {
            s.read((char *) &v, sizeof(float));
            writeWeight(v);
        }
    }

    float *NetworkSerializer::data() {
        return _data.data();
    }

    unsigned int NetworkSerializer::size() const {
        return _data.size();
    }
}
