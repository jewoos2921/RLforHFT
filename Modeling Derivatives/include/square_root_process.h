//
// Created by jewoo on 2025-02-17.
//

#pragma once
namespace MD::Common {
    class SquareRootProcess : public DiffusionProcess {
    public:
        SquareRootProcess(double b, double a,
                          double sigma, double x0 = 0): DiffusionProcess(x0), mean_(b), speed_(a), volatility_(sigma) {
        }

        double drift(Time t, double x) const override {
            return speed_ * (mean_ - x);
        }

        double diffusion(Time t, double x) const override {
            return volatility_ * std::sqrt(x);
        }

    private:
        double mean_, speed_, volatility_;
    };
}
