//
// Created by Renatus Madrigal on 3/3/2025.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "ExtendedDouble.cuh"
#include "MandelbrotSetCuda.h"

#define CHECK_CUDA(err)                                                                                                \
    do {                                                                                                               \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));                                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

namespace Mandelbrot {
    constexpr bool USE_EXTENDED_DOUBLE = true;
    using ComputeDouble = std::conditional_t<USE_EXTENDED_DOUBLE, ExtendedDouble, double>;

    constexpr auto MAX_ITERATIONS = MandelbrotSetCuda::MAX_ITERATIONS;

    __constant__ static uchar3 colors[MAX_ITERATIONS + 1];
    static bool initialized = false;

    void initialize() {
        if (initialized)
            return;
        // TODO: Figure out why this is necessary
        static uchar3 colors_temp[MAX_ITERATIONS + 1];
        initialized = true;
        cv::Mat hsv(1, MAX_ITERATIONS, CV_8UC3);
        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            const double hue = 180 * fmod(n * 0.3, 1.0);
            hsv.at<cv::Vec3b>(0, n) = cv::Vec3b(hue, 255, 255);
        }
        cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);

        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            auto color = hsv.at<cv::Vec3b>(0, n);
            colors_temp[n] = {color[0], color[1], color[2]};
        }
        colors_temp[MAX_ITERATIONS] = {0, 0, 0};

        CHECK_CUDA(cudaMemcpyToSymbol(colors, colors_temp, sizeof(colors_temp)));
    }

    __global__ void mandelbrotKernel(uchar3 *image, // NOLINT
                                     size_t width, size_t height, // NOLINT
                                     double x_min, double x_max, // NOLINT
                                     double y_min, double y_max) {
        constexpr auto ESCAPE_RADIUS_SQ = MandelbrotSetCuda::ESCAPE_RADIUS_SQ;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        const ComputeDouble cr{x_min + (x_max - x_min) * x / width};
        const ComputeDouble ci{y_min + (y_max - y_min) * y / height};

        ComputeDouble zr{0.0}, zi{0.0};
        int n = 0;
        while (n < MAX_ITERATIONS) {
            const ComputeDouble zr2 = zr * zr;
            const ComputeDouble zi2 = zi * zi;
            if (zr2 + zi2 > ESCAPE_RADIUS_SQ)
                break;

            const ComputeDouble zr_temp = zr2 - zi2 + cr;
            zi = zr * zi * 2 + ci;
            zr = zr_temp;
            ++n;
        }
        const int idx = y * width + x;
        image[idx] = colors[n];
    }

    __global__ void mandelbrotKernelWithoutColor(float *image, // NOLINT
                                                 size_t width, size_t height, // NOLINT
                                                 double x_min, double x_max, // NOLINT
                                                 double y_min, double y_max) {
        constexpr auto ESCAPE_RADIUS_SQ = MandelbrotSetCuda::ESCAPE_RADIUS_SQ;
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        const ComputeDouble cr{x_min + (x_max - x_min) * x / width};
        const ComputeDouble ci{y_min + (y_max - y_min) * y / height};

        ComputeDouble zr{0.0}, zi{0.0};
        unsigned int n = 0;
        while (n < MAX_ITERATIONS) {
            const ComputeDouble zr2 = zr * zr;
            const ComputeDouble zi2 = zi * zi;
            if (zr2 + zi2 > ESCAPE_RADIUS_SQ)
                break;

            const ComputeDouble zr_temp = zr2 - zi2 + cr;
            zi = zr * zi * 2 + ci;
            zr = zr_temp;
            ++n;
        }
        const int idx = y * width + x;
        image[idx] = n;
    }

    cv::Mat MandelbrotSetCuda::generateImpl() const {
        initialize();

        uchar3 *d_image;
        CHECK_CUDA(cudaMalloc(&d_image, width_ * height_ * sizeof(uchar3)));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

        mandelbrotKernel<<<grid, block>>>(d_image, width_, height_, x_min_, x_max_, y_min_, y_max_);
        CHECK_CUDA(cudaGetLastError());

        cv::Mat image(height_, width_, CV_8UC3);
        CHECK_CUDA(cudaMemcpy(image.data, d_image, width_ * height_ * sizeof(uchar3), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_image));

        return image;
    }

    cv::Mat MandelbrotSetCuda::generateRawMatrix() const {
        // TODO: reuse memory
        float *d_image;
        CHECK_CUDA(cudaMalloc(&d_image, width_ * height_ * sizeof(int)));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

        mandelbrotKernelWithoutColor<<<grid, block>>>(d_image, width_, height_, x_min_, x_max_, y_min_, y_max_);

        cv::Mat image(height_, width_, CV_32FC1);
        CHECK_CUDA(cudaMemcpy(image.data, d_image, width_ * height_ * sizeof(int), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_image));

        return image;
    }

    cv::Mat MandelbrotSetCuda::detectHighGradient(const cv::Mat &matrix) const {
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


} // namespace Mandelbrot
