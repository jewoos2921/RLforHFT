//
// Created by jewoo on 2025-03-07.
//

#pragma once

#include <iostream>
#include <string>
#include <unordered_set>
#include <sstream>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <fcntl.h>

#include "macros.h"
#include "logging.h"

namespace LL::Common {
    struct SocketCfg {
        std::string ip_;
        std::string iface_;
        int port_ = -1;
        bool is_udp_ = false;
        bool is_listening_ = false;
        bool needs_so_timestamp_ = false;

        auto toString() const {
            std::stringstream ss;
            ss << "SocketCfg[ip: " << ip_ << ", iface: " << iface_ << ", port: " << port_
                    << ", is_udp: " << is_udp_ << ", is_listening: " << is_listening_
                    << ", needs_so_timestamp: " << needs_so_timestamp_ << "]";

            return ss.str();
        }
    };

    constexpr int MaxTCPServerBacklog{1024};

    inline auto getIfaceIP(const std::string &iface) -> std::string {
        char buf[NI_MAXHOST] = {'\0'};
        ifaddrs *ifaddr = nullptr;

        if (getifaddrs(&ifaddr) != -1) {
            for (ifaddrs *ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
                if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET && iface == ifa->ifa_name) {
                    getnameinfo(ifa->ifa_addr,
                                sizeof(sockaddr_in),
                                buf,
                                sizeof(buf),
                                nullptr,
                                0,
                                NI_NUMERICHOST);
                }
            }
            freeifaddrs(ifaddr);
        }
        return buf;
    }

    inline auto setNonBlocking(int fd) -> bool {
        const auto flags = fcntl(fd, F_GETFL, 0);
        if (flags & O_NONBLOCK) {
            return true;
        }
        return fcntl(fd, F_SETFL, flags | O_NONBLOCK) != -1;
    }

    inline auto disableNagle(int fd) -> bool {
        int one = 1;
        return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
                          reinterpret_cast<void *>(&one), sizeof(one)) != -1;
    }

    inline auto setSOTimestamp(int fd) -> bool {
        int one = 1;
        return setsockopt(fd, SOL_SOCKET, SO_TIMESTAMP,
                          reinterpret_cast<void *>(&one), sizeof(one)) != -1;
    }

    inline auto join(int fd, const std::string &ip) -> bool {
        const ip_mreq mreq{
            {inet_addr(ip.c_str())},
            {htonl(INADDR_ANY)}
        };
        return setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                          (&mreq), sizeof(mreq)) != -1;
    }

    [[nodiscard]] inline auto createSocket(Logger& logger,
                                                const SocketCfg &socket_cfg)->int {

    }
}
