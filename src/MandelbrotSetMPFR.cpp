//
// Created by Renatus Madrigal on 3/5/2025.
//

#include "mandelbrot/MandelbrotSetMPFR.h"

#include <iostream>
#include <mpfr.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "mandelbrot/MPFRPool.h"

namespace Mandelbrot {
    constexpr auto PRECISION = 256;
    using Pool = MPFRPool<PRECISION>;
    using MPFRPtr = Pool::MPFRPtr;
    using MPFRObserverPtr = Pool::MPFRPtr::pointer;

    constexpr auto MAX_ITERATIONS = MandelbrotSetMPFR::MAX_ITERATIONS;
    constexpr auto ESCAPE_RADIUS = MandelbrotSetMPFR::ESCAPE_RADIUS;
    constexpr auto ESCAPE_RADIUS_SQ = ESCAPE_RADIUS * ESCAPE_RADIUS;

    size_t computeEscapeTime(MPFRObserverPtr x, MPFRObserverPtr y) {
        static Pool pool;

        // Because OpenMP may convert this to a parallel loop, we need to acquire a new MPFRPtr for each thread instead
        // of using static mpfr_t
        auto zr = pool.acquire();
        auto zi = pool.acquire();
        auto zr2 = pool.acquire();
        auto zi2 = pool.acquire();
        auto temp = pool.acquire();

        // It is guaranteed that the value acquired from the pool is zero by the pool.
        // mpfr_set_zero(*zr, 0);
        // mpfr_set_zero(*zi, 0);

        int n = 0;
        while (n < MAX_ITERATIONS) {
            mpfr_mul(*zr2, *zr, *zr, MPFR_RNDN);
            mpfr_mul(*zi2, *zi, *zi, MPFR_RNDN);

            // Check escape condition
            mpfr_add(*temp, *zr2, *zi2, MPFR_RNDN);
            if (mpfr_cmp_d(*temp, ESCAPE_RADIUS_SQ) > 0) {
                goto exit_computeEscapeTime;
            }

            // temp = zr^2 - zi^2 + x
            mpfr_sub(*temp, *zr2, *zi2, MPFR_RNDN);
            mpfr_add(*temp, *temp, *x, MPFR_RNDN);

            // zi = 2 * zr * zi + y
            mpfr_mul(*zi, *zr, *zi, MPFR_RNDN);
            mpfr_mul_ui(*zi, *zi, 2, MPFR_RNDN);
            mpfr_add(*zi, *zi, *y, MPFR_RNDN);

            // zr = temp
            mpfr_set(*zr, *temp, MPFR_RNDN);

            ++n;
        }

    exit_computeEscapeTime:
        pool.release(std::move(zr));
        pool.release(std::move(zi));
        pool.release(std::move(zr2));
        pool.release(std::move(zi2));
        pool.release(std::move(temp));

        return n;
    }

    cv::Vec3b computeColor(size_t escape_time) {
        static cv::Vec3b colors[MAX_ITERATIONS] = {};
        static bool initialized[MAX_ITERATIONS] = {};
        if (escape_time == MAX_ITERATIONS) {
            return {0, 0, 0};
        }
        if (initialized[escape_time]) {
            return colors[escape_time];
        }

        double hue = 255 * fmod(escape_time * 0.3, 1.0);
        double sat = 255;
        double val = (escape_time < MAX_ITERATIONS) ? 255 : 0;

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        colors[escape_time] = bgr.at<cv::Vec3b>(0, 0);
        initialized[escape_time] = true;
        return bgr.at<cv::Vec3b>(0, 0);
    }

    cv::Mat MandelbrotSetMPFR::generateImpl() const {
        cv::Mat image(height_, width_, CV_8UC3);

        Pool mpfr_pool;

        auto xscale = mpfr_pool.acquire();
        auto yscale = mpfr_pool.acquire();

        mpfr_set_d(*xscale, x_max_ - x_min_, MPFR_RNDN);
        mpfr_div_ui(*xscale, *xscale, width_, MPFR_RNDN);
        mpfr_set_d(*yscale, y_max_ - y_min_, MPFR_RNDN);
        mpfr_div_ui(*yscale, *yscale, height_, MPFR_RNDN);

        auto cur_xscale = mpfr_pool.acquire();
        auto cur_yscale = mpfr_pool.acquire();

        mpfr_set_zero(*cur_yscale, 0);
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (auto y = 0; y < height_; ++y) {
            mpfr_set_zero(*cur_xscale, 0);
            for (auto x = 0; x < width_; ++x) {
                auto xcoord = mpfr_pool.acquire();
                auto ycoord = mpfr_pool.acquire();
                mpfr_set_d(*xcoord, x_min_, MPFR_RNDN);
                mpfr_set_d(*ycoord, y_min_, MPFR_RNDN);
                mpfr_add(*xcoord, *xcoord, *cur_xscale, MPFR_RNDN);
                mpfr_add(*ycoord, *ycoord, *cur_yscale, MPFR_RNDN);

                auto escape_time = computeEscapeTime(xcoord.get(), ycoord.get());
                image.at<cv::Vec3b>(y, x) = computeColor(escape_time);

                mpfr_pool.release(std::move(xcoord));
                mpfr_pool.release(std::move(ycoord));
                mpfr_add(*cur_xscale, *cur_xscale, *xscale, MPFR_RNDN);
            }
            mpfr_add(*cur_yscale, *cur_yscale, *yscale, MPFR_RNDN);
        }

        mpfr_pool.release(std::move(cur_xscale));
        mpfr_pool.release(std::move(cur_yscale));
        mpfr_pool.release(std::move(xscale));
        mpfr_pool.release(std::move(yscale));

        return image;
    }

} // namespace Mandelbrot
