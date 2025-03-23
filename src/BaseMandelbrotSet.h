//
// Created by Renatus Madrigal on 3/3/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H

/**
 * @file BaseMandelbrotSet.h
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace Mandelbrot {
    using ColorSchemeType = cv::Vec3b *;

    constexpr static size_t MAX_ITERATIONS = 1000;
    constexpr static double ESCAPE_RADIUS = 2.0;
    constexpr static double ESCAPE_RADIUS_SQ = ESCAPE_RADIUS * ESCAPE_RADIUS;

    /**
     * @brief The base class for Mandelbrot set.
     * @tparam Derived The derived class.
     */
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

        /**
         * @brief Generate the Mandelbrot set image.
         * @return The Mandelbrot set image.
         * @note The return type is cv::Mat with CV_8UC3.
         */
        [[nodiscard]] cv::Mat generate() const { return colorize(generateRawMatrix()); }

        /**
         * @brief Colorize the matrix.
         * @param matrix The matrix to colorize.
         * @return The colorized image.
         * @note The return type is cv::Mat with CV_8UC3.
         */
        [[nodiscard]] cv::Mat colorize(const cv::Mat &matrix) const {
            assert(colors_);
            cv::Mat image(height_, width_, CV_8UC3);

            for (auto y = 0; y < height_; ++y) {
                for (auto x = 0; x < width_; ++x) {
                    image.at<cv::Vec3b>(y, x) = colors_[static_cast<int>(matrix.template at<float>(y, x))];
                }
            }

            return image;
        }

        /**
         * @brief Generate the raw matrix.
         * @return The raw escape time matrix.
         * @note The return type is cv::Mat with CV_32FC1.
         */
        [[nodiscard]] cv::Mat generateRawMatrix() const {
            return static_cast<const Derived *>(this)->generateRawMatrixImpl();
        }

    protected:
        size_t width_, height_;
        double x_min_, x_max_, y_min_, y_max_;
        ColorSchemeType colors_;
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H
