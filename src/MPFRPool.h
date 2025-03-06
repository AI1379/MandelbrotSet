//
// Created by Renatus Madrigal on 3/5/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_MPFRPOOL_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_MPFRPOOL_H

#include <functional>
#include <memory>
#include <mpfr.h>
#include <mutex>
#include <vector>

namespace Mandelbrot {

    template<size_t Precision>
    class MPFRPool {
        struct Deleter {
            void operator()(mpfr_t *ptr) const {
                mpfr_clear(*ptr);
                free(ptr);
            }
        };

    public:
        using MPFRPtr = std::unique_ptr<mpfr_t, Deleter>;

        explicit MPFRPool(const size_t pool_size = 64) {
            pool_.reserve(pool_size);
            for (auto i = 0u; i < pool_size; ++i) {
                pool_.emplace_back(createMPFR());
            }
        }

        auto acquire() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!pool_.empty()) {
                auto mpfr = std::move(pool_.back());
                pool_.pop_back();
                return mpfr;
            }
            return createMPFR();
        }

        auto release(MPFRPtr mpfr) {
            std::lock_guard<std::mutex> lock(mutex_);
            mpfr_set_zero(*mpfr, 0);
            pool_.emplace_back(std::move(mpfr));
        }

    private:
        std::vector<MPFRPtr> pool_;
        std::mutex mutex_;

        static MPFRPtr createMPFR() {
            // Allocate memory for mpfr_t and initialize it
            MPFRPtr mpfr(static_cast<mpfr_t *>(malloc(sizeof(mpfr_t))), Deleter{});
            mpfr_init2(*mpfr, Precision);
            mpfr_set_zero(*mpfr, 0);
            return mpfr;
        }
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MPFRPOOL_H
