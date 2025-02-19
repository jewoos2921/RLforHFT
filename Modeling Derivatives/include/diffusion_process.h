//
// Created by jewoo on 2025-02-17.
//

#pragma once
namespace MD::Common {
    typedef double Time;
    typedef double Rate;

    class DiffusionProcess {
    public:
        DiffusionProcess(double x0) : x0_(x0) {
        }

        virtual ~DiffusionProcess() = default;

        double x0() const { return x0_; }

        virtual double drift(Time t, double x) const = 0;

        virtual double diffusion(Time t, double x) const = 0;

        virtual double expectation(Time t0, double x0, Time dt) const {
            return x0 + drift(t0, x0) * dt;
        }

        virtual double variance(Time t0, double x0, Time dt) const {
            return diffusion(t0, x0) * diffusion(t0, x0) * dt;
        }

    private:
        double x0_;
    };
}
