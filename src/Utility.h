//
// Created by Renatus Madrigal on 3/11/2025.
//

#ifndef MANDELBROTSET_SRC_UTILITY_H
#define MANDELBROTSET_SRC_UTILITY_H

#if 0

#include <queue>
#include <stdexec/concepts.hpp>
#include <stdexec/coroutine.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

namespace Mandelbrot {
    template<typename T>
    class AsyncChannel {
    public:
        AsyncChannel(ex::scheduler auto from, ex::scheduler auto to) :
            from_(std::move(from)), to_(std::move(to)), queue_{} {}

        ex::sender auto async_send(T value) {
            return ex::just(std::move(value)) // NOLINT
                   | ex::continues_on(from_) // NOLINT
                   | ex::then([this](T val) { queue_.push(std::move(val)); });
        }

        ex::sender auto async_receive() {
            return ex::just() // NOLINT
                   | ex::continues_on(to_) // NOLINT
                   | ex::let_value([this]() {
                         if (queue_.empty()) {
                             return ex::just();
                         } else {
                             T val = queue_.front();
                             queue_.pop();
                             return ex::just(std::move(val));
                         }
                     });
        }

    private:
        std::queue<T> queue_{};
        ex::scheduler auto from_{};
        ex::scheduler auto to_{};
    };

} // namespace Mandelbrot

#endif

#endif // MANDELBROTSET_SRC_UTILITY_H
