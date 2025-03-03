//
// Created by Renatus Madrigal on 3/2/2025.
//

#include "mandelbrot/MandelbrotSet.h"
#include <opencv2/imgproc.hpp>

namespace Mandelbrot {
    MandelbrotSet::MandelbrotSet(size_t width, size_t height) {
        width_ = width;
        height_ = height;
        x_min_ = -2.0;
        x_max_ = 2.0;
        y_min_ = -2.0;
        y_max_ = 2.0;
    }

    MandelbrotSet &MandelbrotSet::setWidth(size_t width) {
        width_ = width;
        return *this;
    }

    MandelbrotSet &MandelbrotSet::setHeight(size_t height) {
        height_ = height;
        return *this;
    }

    MandelbrotSet &MandelbrotSet::setXMin(double x_min) {
        x_min_ = x_min;
        return *this;
    }

    MandelbrotSet &MandelbrotSet::setXMax(double x_max) {
        x_max_ = x_max;
        return *this;
    }

    MandelbrotSet &MandelbrotSet::setYMin(double y_min) {
        y_min_ = y_min;
        return *this;
    }

    MandelbrotSet &MandelbrotSet::setYMax(double y_max) {
        y_max_ = y_max;
        return *this;
    }

    MandelbrotSet &MandelbrotSet::setResolution(size_t width, size_t height) {
        return (this)->setWidth(width).setHeight(height);
    }

    MandelbrotSet &MandelbrotSet::setXRange(double x_min, double x_max) {
        return (this)->setXMin(x_min).setXMax(x_max);
    }

    MandelbrotSet &MandelbrotSet::setYRange(double y_min, double y_max) {
        return (this)->setYMin(y_min).setYMax(y_max);
    }

    size_t MandelbrotSet::getWidth() const { return width_; }
    size_t MandelbrotSet::getHeight() const { return height_; }
    double MandelbrotSet::getXMin() const { return x_min_; }
    double MandelbrotSet::getXMax() const { return x_max_; }
    double MandelbrotSet::getYMin() const { return y_min_; }
    double MandelbrotSet::getYMax() const { return y_max_; }

    size_t MandelbrotSet::compute_escape_time(const std::complex<double> &c) {
        std::complex<double> z(0);
        for (auto i = 0u; i < MAX_ITERATIONS; ++i) {
            z = z * z + c;
            if (std::norm(z) > ESCAPE_RADIUS * ESCAPE_RADIUS) {
                return i;
            }
        }
        return MAX_ITERATIONS;
    }

    cv::Vec3b MandelbrotSet::compute_color(size_t escape_time) {
        if (escape_time == MAX_ITERATIONS) {
            return cv::Vec3b(0, 0, 0);
        }

        double hue = 255 * fmod(escape_time * 0.3, 1.0);
        double sat = 255;
        double val = (escape_time < MAX_ITERATIONS) ? 255 : 0;

        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        return bgr.at<cv::Vec3b>(0, 0);
    }

    cv::Mat MandelbrotSet::generateImage() const {
        cv::Mat image(height_, width_, CV_8UC3);

        const double xscale = (x_max_ - x_min_) / width_;
        const double yscale = (y_max_ - y_min_) / height_;

#pragma omp parallel for
        for (auto y = 0; y < height_; ++y) {
            for (auto x = 0; x < width_; ++x) {
                double xcoord = x_min_ + x * xscale;
                double ycoord = y_min_ + y * yscale;
                std::complex<double> c(xcoord, ycoord);
                size_t escape_time = compute_escape_time(c);
                image.at<cv::Vec3b>(y, x) = compute_color(escape_time);
            }
        }

        return image;
    }


} // namespace Mandelbrot
