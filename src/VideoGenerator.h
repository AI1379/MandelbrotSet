//
// Created by Renatus Madrigal on 3/11/2025.
//

#ifndef MANDELBROTSET_SRC_VIDEOGENERATOR_H
#define MANDELBROTSET_SRC_VIDEOGENERATOR_H

#include <MandelbrotSetCuda.h>
#include <exec/static_thread_pool.hpp>
#include <exec/task.hpp>
#include <opencv2/core.hpp>
#include <stdexec/coroutine.hpp>
#include <stdexec/execution.hpp>
#include "Utility.h"

#include <opencv2/imgproc.hpp>

/**
 * The asynchronous generator flow is as follows:
 * flowchart
 *     KeyFrameGen --> ImgWrite[cv::imwrite]
 *     KeyFrameGen --> TransFrames[Intermediate Frames]
 *     TransFrames --> VideoWrite[cv::VideoWriter.write]
 *     TransFrames --> |Waiting| KeyFrameGen
 */

#if 0

namespace Mandelbrot {
    class VideoGenerator {
    public:
        constexpr static std::string_view key_frame_basename = "frames/MandelbrotSetKeyFrame{}.png";

        using PointType = cv::Point2d;
        using MandelbrotSet = MandelbrotSetCuda;

        // Next center for the Mandelbrot set.
        PointType nextCenter();

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

        VideoGenerator &setXRange(double x_min, double x_max) {
            this->mandelbrot_set_.setXRange(x_min, x_max);
            return *this;
        }

        VideoGenerator &setYRange(double y_min, double y_max) {
            this->mandelbrot_set_.setYRange(y_min, y_max);
            return *this;
        }

        VideoGenerator &setZoomFactor(double zoom_factor) {
            zoom_factor_ = zoom_factor;
            return *this;
        }

        VideoGenerator &setScaleRate(double scale_rate) {
            scale_rate_ = scale_rate;
            return *this;
        }

        VideoGenerator &setMaxStep(size_t max_step) {
            max_step_ = max_step;
            return *this;
        }

        void start() const {}

    private:
        exec::task<void> interpolateFrames() {
            while (current_step_ < max_step_) {
            }
        }

        PointType center_{0.0, 0.0};
        double zoom_factor_{4.0};
        double scale_rate_{1.03};
        size_t max_step_{10};
        size_t current_step_{0};
        exec::static_thread_pool thread_pool_{};
        std::vector<cv::Mat> scale_matrices_{};
        std::vector<cv::Mat> frames_{};
        MandelbrotSet mandelbrot_set_{};
        AsyncChannel<std::pair<cv::Mat, PointType>> frames_channel_{thread_pool_.get_scheduler(),
                                                                    thread_pool_.get_scheduler()};
    };
} // namespace Mandelbrot

#endif

#endif // MANDELBROTSET_SRC_VIDEOGENERATOR_H
