//
// Created by Renatus Madrigal on 3/11/2025.
//

#ifndef MANDELBROTSET_SRC_UTILITY_H
#define MANDELBROTSET_SRC_UTILITY_H

/**
 * @file Utility.h
 * @brief The utility header file.
 */

#include <chrono>
#if __cpp_lib_print >= 202207L
#include <print>
#else
#include <fmt/core.h>
#include <fmt/std.h>
#endif
#include <queue>
#include <ranges>
#include <stdexec/concepts.hpp>
#include <stdexec/coroutine.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

#define TIME_DIFF(start)                                                                                               \
    (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - (start)).count())

namespace Mandelbrot {

    /**
     * @brief The async channel for communication between threads and coroutines.
     * @tparam T The type of the channel.
     * @note The channel is thread-safe.
     */
    template<typename T>
    struct AsyncChannel {

        /**
         * @brief Send the data to the channel.
         * @tparam Args The types of the arguments.
         * @param args The arguments.
         */
        template<typename... Args>
        void send(Args... args) {
            std::lock_guard lock(mutex);
            queue.emplace(std::forward<T>(args)...);
        }

        /**
         * @brief Receive the data from the channel.
         * @return The data.
         * @note This function returns a sender, which is a coroutine. Thus it is awaitable.
         */
        ex::sender auto receive() {
            return ex::just() | ex::then([this]() -> std::optional<T> {
                       if (queue.empty()) {
                           return std::nullopt;
                       } else {
                           T val = queue.front();
                           queue.pop();
                           return val;
                       }
                   });
        }

        /**
         * @brief Check if the channel is empty.
         * @return True if the channel is empty, false otherwise.
         */
        bool empty() const { return queue.empty(); }

    private:
        std::mutex mutex;
        std::queue<T> queue;
    };

    // Unfortunately exec::scope_guard is not working as expected. We need to implement our own.
    /**
     * @brief The scope guard.
     */
    struct ScopeGuard {
        explicit ScopeGuard(std::function<void()> func) : func(std::move(func)) {}
        ~ScopeGuard() { func(); }
        ScopeGuard(const ScopeGuard &) = delete;
        ScopeGuard &operator=(const ScopeGuard &) = delete;

    private:
        std::function<void()> func;
    };

    namespace views = std::views;
    namespace ranges = std::ranges;

#if __cpp_lib_print >= 202207L
    using std::println;
#else
    using fmt::println;
#endif

} // namespace Mandelbrot


#endif // MANDELBROTSET_SRC_UTILITY_H
