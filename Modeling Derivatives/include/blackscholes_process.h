//
// Created by jewoo on 2025-02-17.
//

#pragma once

namespace MD::Common {
    class BlackScholesProcess : public DiffusionProcess {
    public:
        BlackScholesProcess(Rate rate, double volatility,
                            double s0 = 0.0): MD::Common::DiffusionProcess(s0), r_(rate),
                                              sigma_(volatility) {
        }

        double drift(Time t, double x) const override {
            return r_ - 0.5 * sigma_ * sigma_;
        }

        double diffusion(Time t, double x) const override {
            return sigma_;
        }

    private:
        double r_, sigma_;
    };
}
