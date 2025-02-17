//
// Created by jewoo on 2025-02-13.
//

#pragma once

#include <ostream>
#include <istream>
#include <vector>

namespace nnetcpp {
    class NetworkSerializer {
    public :
        NetworkSerializer();

        void writeWeight(float value);

        float readWeight();

        void save(std::ostream &s);

        void load(std::istream &s);

        float *data();

        unsigned int size() const;

    private:
        std::vector<float> _data;
        unsigned int _pos;
    };
}
