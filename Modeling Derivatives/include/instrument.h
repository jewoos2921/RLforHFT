//
// Created by jewoo on 2025-02-17.
//

#pragma once

namespace MD {
    namespace Common {
        class Instrument : public Patterns::Observer, public Patterns::Observable {
        public:
            Instrument(const std::string &isinCode,
                       const std::string &description): NPV_(0.0),
                                                        isExpired_(false), isinCode_(isinCode),
                                                        description_(description), calculated_(false) {
            }

            virtual ~Instrument() = default;

            inline std::string isinCode() const {
                return isinCode_;
            }

            inline std::string description() const {
                return description_;
            }

            inline double NPV() const {
                calculate();
                return isExpired_ ? 0.0 : NPV_;
            }

            inline bool isExpired() const {
                calculate();
                return isExpired_;
            }

            inline void update() {
                calculated_ = false;
                notifyObservers();
            }

        protected:
            virtual void performCalculations() const = 0;

            mutable double NPV_;
            mutable double isExpired_;

        private:
            std::string isinCode_, description_;
            mutable bool calculated_;
        };
    }
}
