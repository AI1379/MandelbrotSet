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
#include <utility>
#include "Algorithm.h"
#include "MandelbrotSet.h"
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

        // Constants for the grid detection
        constexpr static int DIVIDE = 7;
        constexpr static int BLOCK_SIZE = 4;

        unsigned int getWorkerCount() const { return worker_count_; }

        unsigned int getIOCount() const { return io_count_; }

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

        VideoGenerator &setColors(ColorSchemeType colors) {
            mandelbrot_set_.setColors(colors);
            return *this;
        }

        VideoGenerator &autoDetect() {
            auto_detect_ = true;
            return *this;
        }

        VideoGenerator &setAutoDetect(bool auto_detect) {
            auto_detect_ = auto_detect;
            return *this;
        }

        VideoGenerator &showGrid() {
            show_grid_ = true;
            return *this;
        }

        VideoGenerator &setShowGrid(bool show_grid) {
            show_grid_ = show_grid;
            return *this;
        }

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
            transform_matrices_.resize(frame_count_);

            done_ = std::stop_source{};
            exec::async_scope scope;
            auto sched = compute_pool_.get_scheduler();

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

            PointType center = cv::Point2f(mandelbrot_set_.getWidth() / 2.0, mandelbrot_set_.getHeight() / 2.0);

            for (auto [step, factor]: steps) {
                println(stdout, "Generating keyframe {} on thread {} at {}s", step, std::this_thread::get_id(),
                        TIME_DIFF(start_));
                mandelbrot_set_.setCenter(center_.x, center_.y, xsize_ / factor, ysize_ / factor);
                cv::Mat res;
                if (auto_detect_) {
                    auto mat = mandelbrot_set_.generateRawMatrix();
                    auto mask = detectHighGradient(mat);
                    auto l = mat.cols / DIVIDE * (DIVIDE / 2), r = mat.cols / DIVIDE * (DIVIDE / 2 + 1);
                    auto t = mat.rows / DIVIDE * (DIVIDE / 2), b = mat.rows / DIVIDE * (DIVIDE / 2 + 1);
                    auto w = mat.cols / (DIVIDE * BLOCK_SIZE), h = mat.rows / (DIVIDE * BLOCK_SIZE);
                    auto sums = views::cartesian_product(views::iota(0, BLOCK_SIZE), views::iota(0, BLOCK_SIZE)) //
                                | views::transform([&](const auto &p) {
                                      auto [i, j] = p;
                                      return std::make_tuple(cv::Rect(l + i * w, t + j * h, w, h), i, j);
                                  }) //
                                | views::transform([&mask](const auto &p) {
                                      const auto &[rect, i, j] = p;
                                      return std::make_tuple(cv::sum(mask(rect))[0], i, j);
                                  });

                    auto [sum, x, y] =
                            *ranges::max_element(sums, std::less<>(), [](const auto &p) { return std::get<0>(p); });
                    res = mandelbrot_set_.colorize(mat);

                    center = cv::Point2d(l + x * w + w / 2, t + y * h + h / 2);
                    if (show_grid_) {
                        printGrid(res, l, r, t, b, w, h, center);
                    }

                    // Update the center for the next iteration.
                    auto xmin = mandelbrot_set_.getXMin(), xmax = mandelbrot_set_.getXMax();
                    auto ymin = mandelbrot_set_.getYMin(), ymax = mandelbrot_set_.getYMax();
                    center_.x = xmin + center.x * (xmax - xmin) / res.cols;
                    center_.y = ymin + center.y * (ymax - ymin) / res.rows;

                } else {
                    res = mandelbrot_set_.generate();
                }

                println(stdout, "Keyframes generated on thread {} at {}s", std::this_thread::get_id(),
                        TIME_DIFF(start_));

                // Call the interpolation function.
                channel_.send(std::make_pair(res, center));

                // Call the image write function.
                scope.spawn(ex::starts_on(io_pool_.get_scheduler(),
                                          ex::just(std::make_pair(std::move(res), step)) |
                                                  ex::then([this](auto &&arg) { this->imageWrite(arg); })));
            }

            done_.request_stop();
            ex::sync_wait(scope.on_empty());
            println(stdout, "All work done on thread {} at {}s", std::this_thread::get_id(), TIME_DIFF(start_));
        }

    private:
        static void printGrid(cv::Mat &image, int l, int r, int t, int b, int w, int h, cv::Point2d center) {
            cv::line(image, cv::Point(0, t), cv::Point(image.cols, t), cv::Scalar(0, 255, 0), 2);
            cv::line(image, cv::Point(0, b), cv::Point(image.cols, b), cv::Scalar(0, 255, 0), 2);
            cv::line(image, cv::Point(l, 0), cv::Point(l, image.rows), cv::Scalar(0, 255, 0), 2);
            cv::line(image, cv::Point(r, 0), cv::Point(r, image.rows), cv::Scalar(0, 255, 0), 2);

            for (auto k: views::iota(1, BLOCK_SIZE)) {
                cv::line(image, cv::Point(l + k * w, t), cv::Point(l + k * w, b), cv::Scalar(0, 255, 0), 2);
                cv::line(image, cv::Point(l, t + k * h), cv::Point(r, t + k * h), cv::Scalar(0, 255, 0), 2);
            }

            cv::line(image, cv::Point(0, center.y), cv::Point(image.cols, center.y), cv::Scalar(0, 0, 255), 2);
            cv::line(image, cv::Point(center.x, 0), cv::Point(center.x, image.rows), cv::Scalar(0, 0, 255), 2);
            cv::circle(image, center, 30, cv::Scalar(0, 0, 255), 2);
        }

        void computeTransformMatrices(const cv::Point2f &center, double scale_rate, int frames) {
            constexpr double EPS = 1e-9;
            static PointType prev = center;
            static bool initialized = false;
            auto absolute_center = cv::Point2f(mandelbrot_set_.getWidth() / 2.0, mandelbrot_set_.getHeight() / 2.0);
            double factor = 1.0;
            if (std::fabs(center.x - prev.x) > EPS || std::fabs(center.y - prev.y) > EPS || !initialized) {
                auto dx = center.x - absolute_center.x;
                auto dy = center.y - absolute_center.y;
                initialized = true;
                for (int i = 0; i < frames; ++i) {
                    transform_matrices_[i] = getRotationMatrix2D(center, 0, factor);
                    transform_matrices_[i].at<double>(0, 2) -= dx * i / frames;
                    transform_matrices_[i].at<double>(1, 2) -= dy * i / frames;
                    factor *= scale_rate;
                }
                prev = center;
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

            while (!done_.stop_requested() || !channel_.empty()) {
                auto value = co_await channel_.receive();
                if (value) {
                    auto [image, center] = value.value();
                    computeTransformMatrices(center, scale_rate_, frame_count_);

                    println(stdout, "Generating with center: ({}, {}) on thread {} at {}s", center.x, center.y,
                            std::this_thread::get_id(), TIME_DIFF(start_));

                    co_await ( //
                            ex::schedule(compute_pool_.get_scheduler()) //
                            | ex::bulk(frame_count_, [&](int i) {
                                  cv::warpAffine(image, frames_[i], transform_matrices_[i],
                                                 cv::Size(mandelbrot_set_.getWidth(), mandelbrot_set_.getHeight()));
                              }));

                    // for (int i = 0; i < frame_count_; ++i) {
                    //     cv::warpAffine(image, frames_[i], transform_matrices_[i],
                    //                    cv::Size(mandelbrot_set_.getWidth(), mandelbrot_set_.getHeight()));
                    // }

                    // Write the frames to the video.
                    // This has to be synchronous, otherwise the frames will be out of order.
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
        // frame_count = ceil(log(zoom_factor) / log(scale_rate))
        size_t frame_count_{47};
        size_t max_step_{10};
        bool auto_detect_{false};
        bool show_grid_{false};
        std::string video_name_{"MandelbrotSet.mp4"};

        // TODO: Currently the frame basename is hardcoded. We need to make it configurable.
        constexpr static auto frame_basename_ = "frames/MandelbrotSetKeyFrame{}.png";

        // Async settings and buffers
        unsigned int worker_count_{std::thread::hardware_concurrency() / 4 + 1};
        unsigned int io_count_{std::thread::hardware_concurrency() / 2 + 1};
        std::stop_source done_{};
        exec::static_thread_pool compute_pool_{worker_count_};
        exec::static_thread_pool io_pool_{io_count_};
        std::vector<cv::Mat> transform_matrices_{};
        std::vector<cv::Mat> frames_{};
        MandelbrotSetImpl mandelbrot_set_{};
        AsyncChannel<std::pair<cv::Mat, PointType>> channel_{};

        // Time tracking
        std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
    };
} // namespace Mandelbrot

#endif // MANDELBROTSET_SRC_VIDEOGENERATOR_H
