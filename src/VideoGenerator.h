//
// Created by Renatus Madrigal on 3/11/2025.
//

#ifndef MANDELBROTSET_SRC_VIDEOGENERATOR_H
#define MANDELBROTSET_SRC_VIDEOGENERATOR_H

#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/task.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <ranges>
#include <stdexec/coroutine.hpp>
#include <stdexec/execution.hpp>
#include "MandelbrotSetCuda.h"
#include "Utility.h"

/**
 * The asynchronous generator flow is as follows:
 * flowchart
 *     KeyFrameGen --> ImgWrite[cv::imwrite]
 *     KeyFrameGen --> TransFrames[Intermediate Frames]
 *     TransFrames --> VideoWrite[cv::VideoWriter.write]
 *     TransFrames --> |Waiting| KeyFrameGen
 */

namespace Mandelbrot {
    class VideoGenerator {
    public:
        using PointType = cv::Point2d;

#if ENABLE_CUDA
        using MandelbrotSetImpl = Mandelbrot::MandelbrotSetCuda;
#elif ENABLE_MPFR
        using MandelbrotSetImpl = Mandelbrot::MandelbrotSetMPFR;
#else
        using MandelbrotSetImpl = Mandelbrot::MandelbrotSet;
#endif

        // TODO: Make it configurable.
        constexpr static auto worker_count_ = 4;
        constexpr static auto io_count_ = 4;

        // Next center for the Mandelbrot set.
        // TODO: Implement this.
        PointType nextCenter() { return {}; }

        // Settings for video generation
        VideoGenerator &setResolution(size_t width, size_t height) {
            mandelbrot_set_.setResolution(width, height);
            return *this;
        }

        VideoGenerator &setCenter(PointType center) {
            center_ = center;
            return *this;
        }

        VideoGenerator &setCenter(double x_center, double y_center) {
            center_ = PointType(x_center, y_center);
            return *this;
        }

        VideoGenerator &setSize(double xsize, double ysize) {
            xsize_ = xsize;
            ysize_ = ysize;
            return *this;
        }

        VideoGenerator &setZoomFactor(double zoom_factor) {
            zoom_factor_ = zoom_factor;
            return *this;
        }

        VideoGenerator &setScaleRate(double scale_rate) {
            scale_rate_ = scale_rate;
            updateFrameCount();
            return *this;
        }

        VideoGenerator &setMaxStep(size_t max_step) {
            max_step_ = max_step;
            updateFrameCount();
            return *this;
        }

        // TODO: Make this asynchronous.
        void start() {
            println(stdout, "Generating video...");
            println(stdout, "Working threads: {}", worker_count_);
            println(stdout, "IO threads: {}", io_count_);
            println(stdout, "Resolution: {} x {}", mandelbrot_set_.getWidth(), mandelbrot_set_.getHeight());
            println(stdout, "Center: {}, {}", center_.x, center_.y);
            println(stdout, "Initial size: {} x {}", xsize_, ysize_);
            println(stdout, "Max step: {}", max_step_);
            println(stdout, "Zoom factor: {}", zoom_factor_);
            println(stdout, "Scale rate: {}", scale_rate_);
            println(stdout, "Frame count: {}", frame_count_);

            // Start Timer
            start_ = std::chrono::steady_clock::now();
            frames_.resize(frame_count_);
            scale_matrices_.resize(frame_count_);

            precomputeScaleMatrices(center_, scale_rate_, frame_count_);
            done_.store(false);
            exec::async_scope scope;
            auto sched = compute_pool_.get_scheduler();

            MandelbrotSetImpl mandelbrot_set;

            scope.spawn(ex::starts_on(sched, interpolateFrames()));

            // Generate the steps for keyframe generation.
            // It can be easily done by using traditional for loop, but I want to use ranges.
            // Just for fun.
            auto steps = views::zip( //
                    views::iota(0u, max_step_), //
                    views::iota(0u, max_step_) | views::transform([this, factor = 1.0](auto) mutable {
                        auto res = factor;
                        factor *= this->zoom_factor_;
                        return res;
                    }));

            for (auto [step, factor]: steps) {
                println(stdout, "Generating keyframe {} on thread {} at {}s", step, std::this_thread::get_id(),
                        TIME_DIFF(start_));
                mandelbrot_set.setCenter(center_.x, center_.y, xsize_ / factor, ysize_ / factor);
                auto res = mandelbrot_set_.generate();
                println(stdout, "Keyframes generated on thread {} at {}s", std::this_thread::get_id(),
                        TIME_DIFF(start_));
                channel_.send(res);
                scope.spawn(ex::starts_on(io_pool_.get_scheduler(),
                                          ex::just(std::make_pair(std::move(res), step)) |
                                                  ex::then([this](auto &&arg) { this->imageWrite(arg); })));
            }

            done_.store(true);
            ex::sync_wait(scope.on_empty());
            println(stdout, "All work done on thread {} at {}s", std::this_thread::get_id(), TIME_DIFF(start_));
        }

    private:
        void precomputeScaleMatrices(const cv::Point2f &center, double scale_rate, int frames) {
            double factor = 1.0;
            for (int i = 0; i < frames; ++i) {
                scale_matrices_[i] = getRotationMatrix2D(center, 0, factor);
                factor *= scale_rate;
            }
        }

        void updateFrameCount() {
            frame_count_ = static_cast<size_t>(std::ceil(std::log(zoom_factor_) / std::log(scale_rate_)));
        }

        exec::task<void> interpolateFrames() {
            cv::VideoWriter writer;
            ScopeGuard guard{[&]() { writer.release(); }};
            writer.open(video_name_, cv::VideoWriter::fourcc('h', 'v', 'c', 'l'), 30,
                        cv::Size(mandelbrot_set_.getWidth(), mandelbrot_set_.getHeight()), true);
            while (!done_.load() || !channel_.empty()) {
                auto value = co_await channel_.receive();
                if (value) {
                    for (int i = 0; i < frame_count_; ++i) {
                        cv::warpAffine(value.value(), frames_[i], scale_matrices_[i],
                                       cv::Size(mandelbrot_set_.getWidth(), mandelbrot_set_.getHeight()));
                    }
                    for (auto &frame: frames_) {
                        writer.write(frame);
                    }
                    println(stdout, "Video generated on thread {} at {}s", std::this_thread::get_id(),
                            TIME_DIFF(start_));
                    co_await ex::just();
                } else {
                    co_await ex::just();
                }
            }
        }

        void imageWrite(std::pair<cv::Mat, int> &&arg) {
            println(stdout, "Writing image on thread {} at {}s", std::this_thread::get_id(), TIME_DIFF(start_));

            auto &[image, step] = arg;
            auto filename = std::format(frame_basename_, step + 1);
            cv::imwrite(filename, image);

            println(stdout, "Image {} written on thread {} at {}s", filename, std::this_thread::get_id(),
                    TIME_DIFF(start_));
        };

        // Settings for video generation
        PointType center_{0.0, 0.0};
        double xsize_{4.0}, ysize_{4.0};
        double zoom_factor_{4.0};
        double scale_rate_{1.03};
        // ceil(log(zoom_factor) / log(scale_rate))
        size_t frame_count_{47};
        size_t max_step_{10};
        size_t current_step_{0};
        std::string video_name_{"MandelbrotSet.mp4"};
        // TODO: Currently the frame basename is hardcoded. We need to make it configurable.
        constexpr static auto frame_basename_ = "frames/MandelbrotSetKeyFrame{}.png";

        // Async settings and buffers
        // TODO: Custom cancellation token to avoid using bool.
        std::atomic<bool> done_{false};
        exec::static_thread_pool compute_pool_{worker_count_};
        exec::static_thread_pool io_pool_{io_count_};
        std::vector<cv::Mat> scale_matrices_{};
        std::vector<cv::Mat> frames_{};
        MandelbrotSetImpl mandelbrot_set_{};
        AsyncChannel<cv::Mat> channel_{};

        // Time tracking
        std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
    };
} // namespace Mandelbrot

#endif // MANDELBROTSET_SRC_VIDEOGENERATOR_H
