#include "market_data_publisher.h"

namespace LL::Exchange {
    MarketDataPublisher::MarketDataPublisher(MEMarketUpdateLFQueue *market_updates, const std::string &iface,
                                             const std::string &snapshot_ip, int snapshot_port,
                                             const std::string &incremental_ip, int incremental_port)
        : outgoing_md_updates_(market_updates),
          snapshot_md_updates_(ME_MAX_MARKET_UPDATES),
          run_(false),
          logger_("exchange_market_data_publisher.log"),
          incremental_socket_(logger_) {
        ASSERT(incremental_socket_.init(incremental_ip, iface,
                                        incremental_port, false) >= 0,
               "Unable to create incremental mcast socket. error:"
               + std::string(std::strerror(errno)));
        snapshot_synthesizer_ = new SnapshotSynthesizer(&snapshot_md_updates_,
                                                        iface, snapshot_ip, snapshot_port);
    }

    MarketDataPublisher::~MarketDataPublisher() {
        stop();
        using namespace std::literals::chrono_literals;
        std::this_thread::sleep_for(5s);

        delete snapshot_synthesizer_;
        snapshot_synthesizer_ = nullptr;
    }

    auto MarketDataPublisher::start() -> void {
        run_ = true;
        ASSERT(createAndStartThread(-1,
                                    "Exchange/MarketDataPublisher",
                                    [this]() { run(); }) != nullptr,
               "Failed to start MarketDataPublisher");
        snapshot_synthesizer_->start();
    }

    auto MarketDataPublisher::stop() -> void {
        run_ = false;
        snapshot_synthesizer_->stop();
    }

    auto MarketDataPublisher::run() noexcept -> void {
        logger_.log("%:% %() %\n",
                    __FILE__, __LINE__, __FUNCTION__, getCurrentTimeStr(&time_str_));
        while (run_) {
            for (auto market_update = outgoing_md_updates_->getNextToRead();
                 outgoing_md_updates_->size() &&
                 market_update; market_update = outgoing_md_updates_->getNextToRead()) {
                logger_.log("%:% %() % Sending seq:% %\n",
                            __FILE__, __LINE__, __FUNCTION__, getCurrentTimeStr(&time_str_),
                            next_inc_seq_num_, market_update->toString().c_str());

                incremental_socket_.send(&next_inc_seq_num_, sizeof(next_inc_seq_num_));
                incremental_socket_.send(market_update, sizeof(MEMarketUpdate));

                outgoing_md_updates_->updateReadIndex();


                auto next_write = snapshot_md_updates_.getNextToWriteTo();
                next_write->seq_num_ = next_inc_seq_num_;
                next_write->me_market_update_ = *market_update;
                snapshot_md_updates_.updateWriteIndex();

                next_inc_seq_num_++;
            }
            incremental_socket_.sendAndRecv();
        }
    }
}
