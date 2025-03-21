//
// Created by Renatus Madrigal on 3/3/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace Mandelbrot {
    using ColorSchemeType = cv::Vec3b *;

    constexpr static size_t MAX_ITERATIONS = 1000;
    constexpr static double ESCAPE_RADIUS = 2.0;

    template<typename Derived>
    class BaseMandelbrotSet {
    public:
        BaseMandelbrotSet() = default;
        BaseMandelbrotSet(const size_t width, const size_t height) : width_(width), height_(height) {
            x_min_ = y_min_ = -2.0;
            x_max_ = y_max_ = 2.0;
            colors_ = nullptr;
        }

        [[nodiscard]] size_t getWidth() const { return static_cast<const Derived *>(this)->width_; }
        [[nodiscard]] size_t getHeight() const { return static_cast<const Derived *>(this)->height_; }
        [[nodiscard]] double getXMin() const { return static_cast<const Derived *>(this)->x_min_; }
        [[nodiscard]] double getXMax() const { return static_cast<const Derived *>(this)->x_max_; }
        [[nodiscard]] double getYMin() const { return static_cast<const Derived *>(this)->y_min_; }
        [[nodiscard]] double getYMax() const { return static_cast<const Derived *>(this)->y_max_; }

        Derived &setWidth(size_t width) {
            static_cast<Derived *>(this)->width_ = width;
            return *static_cast<Derived *>(this);
        }
        Derived &setHeight(size_t height) {
            static_cast<Derived *>(this)->height_ = height;
            return *static_cast<Derived *>(this);
        }
        Derived &setXMin(double x_min) {
            static_cast<Derived *>(this)->x_min_ = x_min;
            return *static_cast<Derived *>(this);
        }
        Derived &setXMax(double x_max) {
            static_cast<Derived *>(this)->x_max_ = x_max;
            return *static_cast<Derived *>(this);
        }
        Derived &setYMin(double y_min) {
            static_cast<Derived *>(this)->y_min_ = y_min;
            return *static_cast<Derived *>(this);
        }
        Derived &setYMax(double y_max) {
            static_cast<Derived *>(this)->y_max_ = y_max;
            return *static_cast<Derived *>(this);
        }
        Derived &setXRange(double x_min, double x_max) { return setXMin(x_min).setXMax(x_max); }
        Derived &setYRange(double y_min, double y_max) { return setYMin(y_min).setYMax(y_max); }
        Derived &setResolution(size_t width, size_t height) { return setWidth(width).setHeight(height); }

        // xsize is the width of the image in the complex plane, the height is calculated based on the aspect ratio
        Derived &setCenter(double x_center, double y_center, double xsize) {
            auto &self = static_cast<Derived &>(*this);
            auto ysize = xsize * self.height_ / self.width_;
            self.x_min_ = x_center - xsize / 2;
            self.x_max_ = x_center + xsize / 2;
            self.y_min_ = y_center - ysize / 2;
            self.y_max_ = y_center + ysize / 2;
            return self;
        }

        Derived &setCenter(double x_center, double y_center, double xsize, double ysize) {
            auto &self = static_cast<Derived &>(*this);
            self.x_min_ = x_center - xsize / 2;
            self.x_max_ = x_center + xsize / 2;
            self.y_min_ = y_center - ysize / 2;
            self.y_max_ = y_center + ysize / 2;
            return self;
        }

        Derived &setColors(ColorSchemeType colors) {
            static_cast<Derived *>(this)->colors_ = colors;
            return *static_cast<Derived *>(this);
        }

        // generateImpl should return a cv::Mat with CV_8UC3
        [[nodiscard]] cv::Mat generate() const {
            assert(colors_);
            auto mat = generateRawMatrix();
            cv::Mat image(height_, width_, CV_8UC3);

            for (auto y = 0; y < height_; ++y) {
                for (auto x = 0; x < width_; ++x) {
                    image.at<cv::Vec3b>(y, x) = colors_[static_cast<int>(mat.template at<float>(y, x))];
                }
            }

            return image;
        }

        // generateRawMatrixImpl should return cv::Mat with CV_32FC1
        [[nodiscard]] cv::Mat generateRawMatrix() const {
            return static_cast<const Derived *>(this)->generateRawMatrixImpl();
        }

    protected:
        size_t width_, height_;
        double x_min_, x_max_, y_min_, y_max_;
        ColorSchemeType colors_;
    };

    inline cv::Mat detectHighGradient(const cv::Mat &matrix) {
        constexpr static double GRADIENT_THRESHOLD = 0.5;

        cv::Mat normalized;
        cv::normalize(matrix, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        cv::Mat grad_x, grad_y;
        cv::Sobel(normalized, grad_x, CV_32F, 1, 0);
        cv::Sobel(normalized, grad_y, CV_32F, 0, 1);

        cv::Mat abs_grad_x, abs_grad_y, grad_mag;
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_mag);

        double min_val, max_val;
        cv::minMaxLoc(grad_mag, &min_val, &max_val);
        double threshold = min_val + (max_val - min_val) * GRADIENT_THRESHOLD;

        cv::Mat mask = grad_mag;
        cv::threshold(grad_mag, mask, threshold, 255, cv::THRESH_BINARY);
        return mask;
    }

    inline ColorSchemeType colorScheme1() {
        static cv::Vec3b colors[MAX_ITERATIONS + 1] = {};

        for (size_t escape_time = 0; escape_time < MAX_ITERATIONS; ++escape_time) {
            double hue = 255 * fmod(escape_time * 0.3, 1.0);
            double sat = 255;
            double val = 255;

            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, sat, val));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            colors[escape_time] = bgr.at<cv::Vec3b>(0, 0);
        }
        colors[MAX_ITERATIONS] = {0, 0, 0};

        return colors;
    }

    inline ColorSchemeType colorScheme2() {
        static cv::Vec3b colors[MAX_ITERATIONS + 1];

        cv::Mat hsv(1, MAX_ITERATIONS, CV_8UC3);
        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            const double hue = 180 * fmod(n * 0.3, 1.0);
            hsv.at<cv::Vec3b>(0, n) = cv::Vec3b(hue, 255, 255);
        }
        cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            auto color = hsv.at<cv::Vec3b>(0, n);
            colors[n] = {color[0], color[1], color[2]};
        }
        colors[MAX_ITERATIONS] = {0, 0, 0};

        return colors;
    }

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H
