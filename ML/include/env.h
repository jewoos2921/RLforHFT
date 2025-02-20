//
// Created by jewoo on 2025-02-19.
//

#pragma once
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "json.hpp"
#include "serializable.h"

namespace ppo_cpp {
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    class Env : public virtual ISerializable {
    public :
        virtual ~Env() = default;

        virtual std::string get_action_space() = 0;

        virtual std::string get_observation_space() = 0;

        virtual std::string get_action_space_size() = 0;

        virtual std::string get_observation_space_size() = 0;

        virtual int get_num_envs() {
            return 1;
        }

        virtual Mat eset() = 0;

        virtual std::vector<Mat> step(const Mat &actions) = 0;

        virtual void render() = 0;

        virtual float get_time() = 0;

        virtual Mat get_original_obs() = 0;

        virtual Mat get_original_rew() = 0;

    protected:
        template<class T>
        static constexpr const T &clamp(const T &v, const T &lo, const T &hi) {
            return assert(hi != lo), (v < lo) ? lo : (hi < v) ? hi : v;
        }

    public:
        const static std::string SPACE_CONTINOUS;
        const static std::string SPACE_DISCRETE;
    };

    const std::string Env::SPACE_CONTINOUS = "continous";
    const std::string Env::SPACE_DISCRETE = "discrete";
}
