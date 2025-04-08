//
// Created by jewoo on 2025-03-22.
//

#pragma once


#include "types.h"
#include "thread_utils.h"
#include "lf_queue.h"
#include "macros.h"
#include "mcast_socket.h"
#include "mem_pool.h"
#include "logging.h"
#include "market_update.h"
#include "me_order.h"

using namespace LL::Common;

namespace LL::Exchange {
    class SnapshotSynthesizer {
    public:
        SnapshotSynthesizer(MDPMarketUpdateLFQueue *market_updates,
                            const std::string &iface,
                            const std::string &snapshot_ip,
                            int snapshot_port);

        ~SnapshotSynthesizer();

        auto start() -> void;

        auto stop() -> void;

        auto addToSnapshot(const MDPMarketUpdate *market_update);

        auto publishSnapshot();

        auto run() -> void;

        SnapshotSynthesizer() = delete;

        SnapshotSynthesizer(const SnapshotSynthesizer &) = delete;

        SnapshotSynthesizer(const SnapshotSynthesizer &&) = delete;

        auto operator=(const SnapshotSynthesizer &) -> SnapshotSynthesizer & = delete;

        auto operator=(const SnapshotSynthesizer &&) -> SnapshotSynthesizer & = delete;

    private:
        MDPMarketUpdateLFQueue *snapshot_md_updates_ = nullptr;
        Logger logger_;
        volatile bool run_ = false;
        std::string time_str_;
        McastSocket snapshot_socket_;

        std::array<std::array<MEMarketUpdate *, ME_MAX_ORDER_IDS>, ME_MAX_TICKERS> ticker_orders_;
        size_t last_inc_seq_num_{0};
        Nanos last_snapshot_time_{0};

        MemPool<MEMarketUpdate> order_pool_;
    };
}
