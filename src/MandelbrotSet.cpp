//
// Created by Renatus Madrigal on 3/2/2025.
//

#include "mandelbrot/MandelbrotSet.h"
#include <opencv2/imgproc.hpp>
#include "mandelbrot/BaseMandelbrotSet.h"

namespace Mandelbrot {
    size_t MandelbrotSet::computeEscapeTime(const std::complex<double> &c) {
        std::complex<double> z(0);
        for (auto i = 0u; i < MAX_ITERATIONS; ++i) {
            z = z * z + c;
            if (std::norm(z) > ESCAPE_RADIUS * ESCAPE_RADIUS) {
                return i;
            }
        }
        return MAX_ITERATIONS;
    }

    cv::Vec3b MandelbrotSet::computeColor(size_t escape_time) {
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

    cv::Mat MandelbrotSet::generateImpl() const {
        cv::Mat image(height_, width_, CV_8UC3);

        const double xscale = (x_max_ - x_min_) / width_;
        const double yscale = (y_max_ - y_min_) / height_;

#pragma omp parallel for
        for (auto y = 0; y < height_; ++y) {
            for (auto x = 0; x < width_; ++x) {
                double xcoord = x_min_ + x * xscale;
                double ycoord = y_min_ + y * yscale;
                std::complex<double> c(xcoord, ycoord);
                size_t escape_time = computeEscapeTime(c);
                image.at<cv::Vec3b>(y, x) = computeColor(escape_time);
            }
        }

        return image;
    }


} // namespace Mandelbrot
