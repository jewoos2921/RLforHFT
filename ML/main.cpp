#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <algorithm>

#include "flappy_simulator.h"
#include "sarsa.h"
#include "epsilon_greedy.h"
#include "tile_coding.h"


namespace rlagent {
    char *get_cmd_option(char **begin, char **end, const std::string &option) {
        char **iter = std::find(begin, end, option);
        if (iter != end && ++iter != end) {
            return *iter;
        }
        return nullptr;
    }

    bool cmd_option_exists(char **begin, char **end, const std::string &option) {
        return std::find(begin, end, option) != end;
    }

    void learn_batch(Learner *learner,
                     int batch_size, int episode_length,
                     std::vector<double> &msve_batch, std::vector<double> &reward_batch) {
        std::cout << "Batch size: " << batch_size << std::endl;

        learner->learn(batch_size, episode_length, msve_batch, reward_batch);

        std::cout << "batch mean reward: " <<
                std::accumulate(reward_batch.begin(), reward_batch.end(), 0.0) / reward_batch.size() << std::endl <<
                "batch msve: "
                << std::accumulate(msve_batch.begin(), msve_batch.end(), 0.0) / msve_batch.size() << std::endl;
    }

    void save_statistics(std::string filename,
                         const double *msve_array,
                         const double *reward_array,
                         int number_of_episodes) {
        std::fstream stats;
        stats.open(filename, std::ios_base::app | std::ios_base::ate);
        if (!stats.tellp()) {
            std::cout << "Create file: " << filename << std::endl;
            stats << "episode,msve,reward" << std::endl;
        }
        std::cout << "Write into file: " << filename << std::endl;
        for (int i = 0; i < number_of_episodes; i++) {
            stats << i << "," << msve_array[i] << "," << reward_array[i] << std::endl;
        }
        stats.close();
    }
}

using namespace rlagent;

int main(int argc, char **argv) {
    const char *execution_mode = get_cmd_option(
        argv, argv + argc, "-exec");

    bool mode_learn{false};
    bool mode_play{false};
    if (execution_mode) {
        mode_learn = std::string(execution_mode) == 'learn';
        mode_play = std::string(execution_mode) == "play";
    }

    const char *working_directory = get_cmd_option(
        argv, argv + argc, "-wdir");
    if (!working_directory) {
        working_directory = "./";
    }

    FlappySimulator env(true);


    const double learning_rate{1e-1};
    const int tilings = 5;
    auto displacement = (Eigen::Matrix<int, 5, 1>() << 1, 3, 5, 7, 11).finished();
    auto state_space_segments = (Eigen::Matrix<int, 5, 1>() << 10, 10, 10, 10, 10).finished();
    auto state_space_min = (Eigen::Matrix<float, 5, 1>() << 0, 3.75, 3.75, 1, -10).finished();
    auto state_space_max = (Eigen::Matrix<float, 5, 1>() << 11, 10.25, 10.25, 13, 10).finished();
    auto approximator = std::make_shared<TileCoding>(
        env.getNumberOfActions(),
        env.getStateDim(),
        learning_rate,
        tilings,
        displacement,
        state_space_segments,
        state_space_min,
        state_space_max);

    double epsilon = 0.2;
    double epsilon_decay = 1.0 - 3e-6;
    auto policy = std::make_shared<EpsilonGreedy>(epsilon, approximator);

    const int number_of_episodes = 1e6;
    Learner::reward_function reward = [](VectorXD x, int a, VectorXD x_next, Environment *env) {
        auto *flappy_env = (FlappySimulator *) env;
        return flappy_env->getCollision() ? -100.0 : 1.0;
    };
    Learner::environment_function init_env = []() {
        auto env = std::make_shared<FlappySimulator>();
        return env;
    };
    const double discount{0.9};
    Sarsa learner(discount, policy, approximator, reward, init_env, 20);

    if (mode_play) {
        approximator->load(std::string(working_directory) + "/approximator.dat");

        policy->_epsilon = policy->_epsilon * std::pow(epsilon_decay, number_of_episodes);

        env.play(policy, 300.0);
    }

    if (mode_learn) {
        const int episode_length = 400;
        learner._verbose = true;

        const int number_of_batches = 100;
        int current_batch = 0;
        int remaining_episodes = number_of_episodes;
        while (remaining_episodes > 0) {
            std::cout << "Batch number: " << ++current_batch << std::endl;

            int batch_size = std::min(
                std::ceil(double(number_of_episodes) / number_of_batches),
                double(remaining_episodes));

            std::vector<double> msve_batch, reward_batch;
            learn_batch(&learner, batch_size, episode_length, msve_batch, reward_batch);
            approximator->save(std::string(working_directory) + "/approximator.dat");
            save_statistics(std::string(working_directory) + "/statistics.csv",
                            msve_batch.data(), reward_batch.data(), batch_size);

            env.play(policy, 10.0);

            policy->_epsilon = policy->_epsilon * std::pow(epsilon_decay, batch_size);
            remaining_episodes -= batch_size;
        }
    }
    std::cout << "Finished " << std::endl;
    return 0;
}
