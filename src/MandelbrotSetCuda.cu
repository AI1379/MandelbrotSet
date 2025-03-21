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

    cv::Mat MandelbrotSetCuda::generateRawMatrixImpl() const {
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

} // namespace Mandelbrot
