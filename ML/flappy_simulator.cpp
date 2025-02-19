//
// Created by jewoo on 2025-02-18.
//
#include "flappy_simulator.h"


namespace rlagent {
    void FlappySimulator::randomizeSeed() {
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        srand((unsigned int) us);
    }

    bool FlappySimulator::checkOverlap(double R, double Xc, double Yc, double X1, double y1, double X2, double Y2) {
        double Xn = std::max(X1, std::min(Xc, X2));
        double Yn = std::max(y1, std::min(Yc, Y2));

        double Dx = Xn - Xc;
        double Dy = Yn - Yc;
        return (Dx * Dx + Dy * Dy) <= R * R;
    }

    FlappySimulator::FlappySimulator(bool with_gui) : Environment() {
        _state = VectorXD::Zero(SIZE_OF_STATESPACE);
        Environment::reset();

        if (!with_gui) {
            _window = nullptr;
            return;
        }

        SDL_Init(SDL_INIT_VIDEO);
        _window = SDL_CreateWindow(
            "Flappy Simulator",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            screen_width * screen_scale,
            screen_height * screen_scale,
            0
        );

        if (!_window) {
            std::cout << "Failed to create window" << std::endl << "SDL error: " << SDL_GetError() << std::endl;
        }

        _window_surface = SDL_GetWindowSurface(_window);
        if (!_window_surface) {
            std::cout << "Failed to get window's surface" << std::endl << "SDL error: " << SDL_GetError() << std::endl;
        }
    }

    FlappySimulator::~FlappySimulator() {
        if (_window) {
            SDL_FreeSurface(_window_surface);
            SDL_DestroyWindow(_window);
            SDL_Quit();
        }
    }

    void FlappySimulator::render(std::string mode) {
        if (_window) {
            std::cout << "Can only render in GUI mode" << std::endl;
            return;
        }

        SDL_FillRect(_window_surface, nullptr, SDL_MapRGB(_window_surface->format, 0, 0, 0));

        SDL_Rect flappy_position;
        flappy_position.x = (flappy_x - flappy_radius) * screen_scale;
        flappy_position.y = (_state[FLAPPY_Y] - flappy_radius) * screen_scale;
        flappy_position.w = flappy_radius * 2 * screen_scale;
        flappy_position.h = flappy_radius * 2 * screen_scale;
        SDL_FillRect(_window_surface, &flappy_position,
                     SDL_MapRGB(_window_surface->format, 200, 0, 0));

        double pipe_x = _state[PIPE_1_X];
        for (int i = 0; i < 2; i++) {
            double pipe_left = pipe_x - pipe_width;
            double pipe_lower_top = _state[PIPE_1_Y + i] + pipe_opening / 2.0;
            double pipe_lower_bottom = screen_height;
            double pipe_upper_top = 0.0;
            double pipe_upper_bottom = pipe_lower_top - pipe_opening;

            SDL_Rect pipe_lower_position;
            pipe_lower_position.x = pipe_left * screen_scale;
            pipe_lower_position.y = pipe_lower_top * screen_scale;
            pipe_lower_position.w = pipe_width * screen_scale;
            pipe_lower_position.h = (pipe_lower_bottom - pipe_lower_top) * screen_scale;
            SDL_FillRect(_window_surface, &pipe_lower_position,
                         SDL_MapRGB(_window_surface->format, 0, 0, 200));

            SDL_Rect pipe_upper_position;
            pipe_upper_position.x = pipe_left * screen_scale;
            pipe_upper_position.y = pipe_upper_top * screen_scale;
            pipe_upper_position.w = pipe_width * screen_scale;
            pipe_upper_position.h = (pipe_upper_bottom - pipe_upper_top) * screen_scale;
            SDL_FillRect(_window_surface, &pipe_upper_position,
                         SDL_MapRGB(_window_surface->format, 0, 0, 200));

            pipe_x += pipe_distance;
            if (pipe_x > screen_width + pipe_width)
                pipe_x -= (pipe_distance * 2);
        }

        SDL_UpdateWindowSurface(_window);
    }

    void FlappySimulator::play(std::shared_ptr<Policy> policy, double play_time_sec, double speedup) {
        if (_window) {
            std::cout << "Can only play in GUI mode" << std::endl;
            return;
        }

        Environment::reset();

        auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
        auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        auto frame_ms = start_ms - dt * 1000.0 / speedup;

        int selected_action = -1;
        bool keep_running = true;

        while (keep_running) {
            SDL_Event e;
            while (SDL_PollEvent(&e) > 0) {
                switch (e.type) {
                    case SDL_QUIT:
                        keep_running = false;
                        break;
                }
            }

            unsigned char const *keys = SDL_GetKeyboardState(nullptr);
            if (keys[SDL_SCANCODE_LSHIFT]) {
                selected_action = keys[SDL_SCANCODE_SPACE] ? 1 : 0;
            }

            now = std::chrono::high_resolution_clock::now().time_since_epoch();
            auto current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

            if (current_ms - frame_ms >= dt * 1000.0 / speedup) {
                VectorXD obs = getState();
                bool episode_over = false;

                if (selected_action < 0)
                    selected_action = policy->apply(getState());

                step(selected_action, obs, episode_over);
                render();

                if (episode_over) reset(obs);
                frame_ms = current_ms;
                selected_action = -1;
            }

            keep_running &= (current_ms - start_ms < play_time_sec * 1000.0);
        }
    }

    void FlappySimulator::step(int action, Eigen::Ref<VectorXD> observation, bool &done) {
        double acceleration = gravity - double(action) * flappy_accel;
        double flappy_v = _state[FLAPPY_V] + acceleration * dt;

        if (std::abs(flappy_v) > flappy_vmax) {
            flappy_v = flappy_v / std::abs(flappy_v) * flappy_vmax;
        }

        double flappy_y = _state[FLAPPY_Y] + _state[FLAPPY_V] * dt + 0.5 * acceleration * std::pow(dt, 2.0);

        double pipe_1_x = _state[PIPE_1_X] - flappy_speed + dt;
        double pipe_1_y = _state[PIPE_1_Y];
        double pipe_2_y = _state[PIPE_2_Y];
        if (pipe_1_x < 0.0) {
            pipe_1_x += pipe_distance * 2.0;
            pipe_1_y = double(rand()) / double(RAND_MAX) * (screen_height - pipe_opening * 1.5) + pipe_opening * 0.75;
        }
        if (pipe_1_x - pipe_distance < 0.0 && pipe_1_x - pipe_distance >= -flappy_speed * dt) {
            pipe_2_y = double(rand()) / double(RAND_MAX) * (screen_height - pipe_opening * 1.5) + pipe_opening * 0.75;
        }

        _collison = false;
        double pipe_x = pipe_1_x;
        for (int i = 0; i < 2; ++i) {
            double pipe_right = pipe_x;
            double pipe_left = pipe_x - pipe_width;
            double pipe_lower_top = _state[PIPE_1_Y + i] + pipe_opening / 2.0;
            double pipe_lower_bottom = screen_height;
            double pipe_upper_top = 0.0;
            double pipe_upper_bottom = pipe_lower_top - pipe_opening;

            _collison = _collison || checkOverlap(flappy_radius, flappy_x, flappy_y, pipe_left, pipe_lower_top,
                                                  pipe_right, pipe_lower_bottom);
            _collison = _collison || checkOverlap(flappy_radius, flappy_x, flappy_y, pipe_left, pipe_upper_top,
                                                  pipe_right, pipe_upper_bottom);

            pipe_x += pipe_distance;
            if (pipe_x > screen_width + pipe_width)
                pipe_x -= (pipe_distance * 2);
        }

        _collison = _collison || (flappy_y + flappy_radius >= screen_height);
        _collison = _collison || (flappy_y - flappy_radius <= 0.0);
        if (_collison) {
            flappy_y = _state[FLAPPY_Y];
            flappy_v = 0.0;
        }

        done = _collison;

        _state[PIPE_1_X] = pipe_1_x;
        _state[PIPE_1_Y] = pipe_1_y;
        _state[PIPE_2_Y] = pipe_2_y;
        _state[FLAPPY_V] = flappy_v;
        _state[FLAPPY_Y] = flappy_y;

        observation = _state;
    }

    void FlappySimulator::reset(Eigen::Ref<VectorXD> observation) {
        randomizeSeed();
        _state = (Eigen::MatrixXd::Random(SIZE_OF_STATESPACE, 1).array() + 1.0) / 2.0;
        _state[PIPE_1_X] = flappy_x - flappy_radius + double(rand() % 2) * pipe_distance;
        _state[PIPE_1_Y] = _state[PIPE_1_Y] * (screen_height - pipe_opening * 1.5) + pipe_opening * 0.75;
        _state[PIPE_2_Y] = _state[PIPE_2_Y] * (screen_height - pipe_opening * 1.5) + pipe_opening * 0.75;

        _state[FLAPPY_Y] = screen_height / 2.0;
        _state[FLAPPY_V] = 0.0;
        _collison = false;
        observation = _state;
    }
}
