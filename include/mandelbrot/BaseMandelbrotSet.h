//
// Created by Renatus Madrigal on 3/3/2025.
//

#ifndef MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H
#define MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H

#include <concepts>
#include <type_traits>

namespace Mandelbrot {
    template<typename Derived>
    class BaseMandelbrotSet {
    public:
        friend class BaseMandelbrotSet<Derived>;

        BaseMandelbrotSet() = default;
        BaseMandelbrotSet(const size_t width, const size_t height) : width_(width), height_(height) {
            x_min_ = y_min_ = -2.0;
            x_max_ = y_max_ = 2.0;
        }

        [[nodiscard]] size_t getWidth(this Derived &self) { return self.width_; }
        [[nodiscard]] size_t getHeight(this Derived &self) { return self.height_; }
        [[nodiscard]] double getXMin(this Derived &self) { return self.x_min_; }
        [[nodiscard]] double getXMax(this Derived &self) { return self.x_max_; }
        [[nodiscard]] double getYMin(this Derived &self) { return self.y_min_; }
        [[nodiscard]] double getYMax(this Derived &self) { return self.y_max_; }

        Derived &setWidth(this Derived &self, size_t width) {
            self.width_ = width;
            return self;
        }
        Derived &setHeight(this Derived &self, size_t height) {
            self.height_ = height;
            return self;
        }
        Derived &setXMin(this Derived &self, double x_min) {
            self.x_min_ = x_min;
            return self;
        }
        Derived &setXMax(this Derived &self, double x_max) {
            self.x_max_ = x_max;
            return self;
        }
        Derived &setYMin(this Derived &self, double y_min) {
            self.y_min_ = y_min;
            return self;
        }
        Derived &setYMax(this Derived &self, double y_max) {
            self.y_max_ = y_max;
            return self;
        }
        Derived &setXRange(this Derived &self, double x_min, double x_max) {
            return self.setXMin(x_min).setXMax(x_max);
        }
        Derived &setYRange(this Derived &self, double y_min, double y_max) {
            return self.setYMin(y_min).setYMax(y_max);
        }
        Derived &setResolution(this Derived &self, size_t width, size_t height) {
            return self.setWidth(width).setHeight(height);
        }

        [[nodiscard]] cv::Mat generate(this Derived &self) { return self.generateImpl(); }

    protected:
        size_t width_, height_;
        double x_min_, x_max_, y_min_, y_max_;
    };

} // namespace Mandelbrot

#endif // MANDELBROTSET_INCLUDE_MANDELBROT_BASEMANDELBROTSET_H
