//
// Created by jewoo on 2025-03-22.
//

#include "snapshot_synthesizer.h"

namespace LL::Exchange {
    SnapshotSynthesizer::SnapshotSynthesizer(MDPMarketUpdateLFQueue *market_updates, const std::string &iface,
                                             const std::string &snapshot_ip, int snapshot_port)
        : snapshot_md_updates_(market_updates),
          logger_("exchange_snapshot_synthesizer.log"), snapshot_socket_(logger_),
          order_pool_(ME_MAX_ORDER_IDS) {
        ASSERT(snapshot_socket_.init(snapshot_ip, iface, snapshot_port, false) >= 0,
               "Unable to create snapshot mcast socket. error:" + std::string(std::strerror(errno)));
        for (auto &orders: ticker_orders_)
            orders.fill(nullptr);
    }

    SnapshotSynthesizer::~SnapshotSynthesizer() {
        stop();
    }

    auto SnapshotSynthesizer::start() -> void {
        run_ = true;
        ASSERT(createAndStartThread(-1, "Exchange/SnapshotSynthesizer",
                                    [this]() { run(); }) != nullptr, "Failed to start SnapshotSynthesizer thread.");
    }

    auto SnapshotSynthesizer::stop() -> void {
        run_ = false;
    }

    auto SnapshotSynthesizer::addToSnapshot(const MDPMarketUpdate *market_update) {
        const auto &me_market_update = market_update->me_market_update_;
        auto *orders = &ticker_orders_.at(me_market_update.ticker_id_);
        switch (me_market_update.type_) {
            case MarketUpdateType::ADD: {
                auto order = orders->at(me_market_update.order_id_);
                ASSERT(order == nullptr, "Received:" + me_market_update.toString()
                                         + " but order already exists: " + (order ? order->toString() : ""));
                orders->at(me_market_update.order_id_) = order_pool_.allocate(me_market_update);
            }
            break;
            case MarketUpdateType::MODIFY: {
                auto order = orders->at(me_market_update.order_id_);
                ASSERT(order != nullptr, "Received:" + me_market_update.toString()
                                         + " but order does not exist.");
                ASSERT(order->order_id_ == me_market_update.order_id_,
                       "Expecting existing order to match new one.");
                ASSERT(order->side_ == me_market_update.side_,
                       "Expecting existing order to match new one.");

                order->quantity_ = me_market_update.quantity_;
                order->price_ = me_market_update.price_;
            }

            break;
            case MarketUpdateType::CANCEL: {
                auto order = orders->at(me_market_update.order_id_);
                ASSERT(order != nullptr, "Received:" + me_market_update.toString()
                                         + " but order does not exist.");
                ASSERT(order->order_id_ == me_market_update.order_id_,
                       "Expecting existing order to match new one.");
                ASSERT(order->side_ == me_market_update.side_,
                       "Expecting existing order to match new one.");

                order_pool_.deallocate(order);
                orders->at(me_market_update.order_id_) = nullptr;
            }
            break;

            case MarketUpdateType::SNAPSHOT_START:
            case MarketUpdateType::CLEAR:
            case MarketUpdateType::SNAPSHOT_END:
            case MarketUpdateType::TRADE:
            case MarketUpdateType::INVALID:
                break;
        }

        ASSERT(market_update->seq_num_ == last_inc_seq_num_ + 1,
               "Expected incremental seq_nums to increase.");
        last_inc_seq_num_ = market_update->seq_num_;
    }

    auto SnapshotSynthesizer::publishSnapshot() {
    }

    auto SnapshotSynthesizer::run() -> void {
    }
}
