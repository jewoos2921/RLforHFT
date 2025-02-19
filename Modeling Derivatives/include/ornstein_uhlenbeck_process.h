//
// Created by jewoo on 2025-02-17.
//


#pragma once

namespace MD::Common {
    class OrnsteinUhlenbeckProcess : public DiffusionProcess {
    public:
        OrnsteinUhlenbeckProcess(double speed, double vol, double x0 = 0.0)
            : DiffusionProcess(x0), speed_(speed), volatility_(vol) {
        }

        double drift(Time t, double x) const override {
            return -speed_ * x;
        }

        double diffusion(Time t, double x) const override {
            return volatility_;
        }

        double expectation(Time t0, double x0, Time dt) const {
            return x0 * exp(-speed_ * dt);
        }

    private:
        double speed_, volatility_;
    };
}
