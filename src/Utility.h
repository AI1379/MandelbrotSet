//
// Created by Renatus Madrigal on 3/11/2025.
//

#ifndef MANDELBROTSET_SRC_UTILITY_H
#define MANDELBROTSET_SRC_UTILITY_H

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

    template<typename T>
    struct AsyncChannel {
        template<typename... Args>
        void send(Args... args) {
            std::lock_guard lock(mutex);
            queue.emplace(std::forward<T>(args)...);
        }

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

        bool empty() const { return queue.empty(); }

    private:
        std::mutex mutex;
        std::queue<T> queue;
    };

    // Unfortunately exec::scope_guard is not working as expected. We need to implement our own.
    struct ScopeGuard {
        explicit ScopeGuard(std::function<void()> func) : func(std::move(func)) {}
        ~ScopeGuard() { func(); }
        ScopeGuard(const ScopeGuard &) = delete;
        ScopeGuard &operator=(const ScopeGuard &) = delete;

    private:
        std::function<void()> func;
    };

    namespace views = std::views;

#if __cpp_lib_print >= 202207L
    using std::println;
#else
    using fmt::println;
#endif

} // namespace Mandelbrot


#endif // MANDELBROTSET_SRC_UTILITY_H
