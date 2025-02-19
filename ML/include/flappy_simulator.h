//
// Created by jewoo on 2025-02-18.
//

#pragma once
#include "environment.h"
#include "policy.h"
#include "subprojects/sdl/include/SDL.h"
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace rlagent {
    class FlappySimulator : public Environment {
    private:
        SDL_Window *_window;
        SDL_Surface *_window_surface;

        VectorXD _state;
        bool _collison;

        void randomizeSeed();

        bool checkOverlap(double R, double Xc, double Yc,
                          double X1, double y1, double X2, double Y2);

    public:
        static const int SIZE_OF_STATESPACE = 5;
        static const int PIPE_1_X = 0;
        static const int PIPE_1_Y = 1;
        static const int PIPE_2_Y = 2;
        static const int FLAPPY_Y = 3;
        static const int FLAPPY_V = 4;

        static constexpr double screen_scale = 30.0;
        static constexpr double screen_width = 9.0;
        static constexpr double screen_height = 14.0;
        static constexpr double pipe_distance = 6.0;
        static constexpr double pipe_opening = 5.0;
        static constexpr double pipe_width = 2.0;
        static constexpr double flappy_radius = 1.0;
        static constexpr double flappy_accel = 100.0;
        static constexpr double flappy_speed = 2.0;
        static constexpr double flappy_vmax = 10.0;
        static constexpr double flappy_x = 4.0;
        static constexpr double gravity = 9.81;
        static constexpr double dt = 1.0 / 20.0;

        FlappySimulator(bool with_gui = false);

        ~FlappySimulator();

        void render(std::string mode = "graphic") override;

        void play(std::shared_ptr<Policy> policy,
                  double play_time_sec = 10.0,
                  double speedup = 1.0);

        int getNumberOfActions() override { return 2; }
        int getStateDim() override { return SIZE_OF_STATESPACE; }
        VectorXD getState() override { return _state; }
        bool getCollision() { return _collison; }

        void step(int action, Eigen::Ref<VectorXD> observation, bool &done) override;

        void reset(Eigen::Ref<VectorXD> observation) override;
    };
}
