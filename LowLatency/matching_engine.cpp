//
// Created by jewoo on 2025-03-21.
//

#include "matching_engine.h"

namespace LL::Exchange {
    MatchingEngine::MatchingEngine(ClientRequestLFQueue *client_requests, ClientResponseLFQueue *client_responses,
                                   MEMarketUpdateLFQueue *market_updates)
        : incoming_requests_(client_requests),
          outgoing_ogw_responses_(client_responses),
          outgoing_md_updates_(market_updates),
          logger_("exchange_matching_engine.log") {
        for (size_t i = 0; i < ticker_order_books_.size(); ++i) {
            ticker_order_books_[i] = new MEOrderBook(i, &logger_, this);
        }
    }

    MatchingEngine::~MatchingEngine() {
        run_ = false;

        using namespace std::literals::chrono_literals;
        std::this_thread::sleep_for(1s);

        incoming_requests_ = nullptr;
        outgoing_ogw_responses_ = nullptr;
        outgoing_md_updates_ = nullptr;

        for (auto &order_book: ticker_order_books_) {
            delete order_book;
            order_book = nullptr;
        }
    }

    auto MatchingEngine::start() -> void {
        run_ = true;
        ASSERT(createAndStartThread(-1,
                                    "Exchange/MatchingEngine",
                                    [this]() {
                                        run();
                                    }) != nullptr, "Failed to start MatchingEngine::start()");
    }

    auto MatchingEngine::stop() -> void {
        run_ = false;
    }
}
