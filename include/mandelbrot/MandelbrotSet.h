//
// Created by Renatus Madrigal on 3/2/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H

#include <complex>
#include <opencv2/core/mat.hpp>

namespace Mandelbrot {

    class MandelbrotSet {
    public:
        constexpr static size_t MAX_ITERATIONS = 1000;
        constexpr static double ESCAPE_RADIUS = 2.0;
        constexpr static size_t COLOR_CYCLE = 50;

        MandelbrotSet() = default;
        MandelbrotSet(size_t width, size_t height);

        MandelbrotSet &setWidth(size_t width);
        MandelbrotSet &setHeight(size_t height);
        MandelbrotSet &setXMin(double x_min);
        MandelbrotSet &setXMax(double x_max);
        MandelbrotSet &setYMin(double y_min);
        MandelbrotSet &setYMax(double y_max);
        MandelbrotSet &setResolution(size_t width, size_t height);
        MandelbrotSet &setXRange(double x_min, double x_max);
        MandelbrotSet &setYRange(double y_min, double y_max);

        [[nodiscard]] size_t getWidth() const;
        [[nodiscard]] size_t getHeight() const;
        [[nodiscard]] double getXMin() const;
        [[nodiscard]] double getXMax() const;
        [[nodiscard]] double getYMin() const;
        [[nodiscard]] double getYMax() const;

        cv::Mat generateImage() const;

#ifdef MANDELBROT_SET_TEST
        // TODO: Add test helper.
        friend class MandelbrotSetTest;
#endif

    private:
        [[nodiscard]] static size_t compute_escape_time(const std::complex<double> &c);

        [[nodiscard]] static cv::Vec3b compute_color(size_t escape_time);

        size_t width_, height_;
        double x_min_, x_max_, y_min_, y_max_;
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_MANDELBROTSET_H
