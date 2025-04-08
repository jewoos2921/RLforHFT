//
// Created by jewoo on 2025-03-22.
//

#pragma once


#include <functional>
#include "snapshot_synthesizer.h"

namespace LL::Exchange {
    class MarketDataPublisher {
    public:
        MarketDataPublisher(MEMarketUpdateLFQueue *market_updates,
                            const std::string &iface,
                            const std::string &snapshot_ip,
                            int snapshot_port,
                            const std::string &incremental_ip,
                            int incremental_port);

        ~MarketDataPublisher();

        auto start() -> void;

        auto stop() -> void;

        auto run() noexcept -> void;

        MarketDataPublisher() = delete;

        MarketDataPublisher(const MarketDataPublisher &) = delete;

        MarketDataPublisher(const MarketDataPublisher &&) = delete;

        auto operator=(const MarketDataPublisher &) -> MarketDataPublisher & = delete;

        auto operator=(const MarketDataPublisher &&) -> MarketDataPublisher & = delete;

    private:
        size_t next_inc_seq_num_{1};
        MEMarketUpdateLFQueue *outgoing_md_updates_ = nullptr;
        MDPMarketUpdateLFQueue snapshot_md_updates_;

        volatile bool run_{false};
        std::string time_str_;
        Logger logger_;

        McastSocket incremental_socket_;
        SnapshotSynthesizer *snapshot_synthesizer_ = nullptr;
    };
}
