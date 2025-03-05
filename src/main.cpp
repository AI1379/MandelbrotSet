//
// Created by Renatus Madrigal on 3/2/2025.
//

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "mandelbrot/MandelbrotSet.h"
#include "mandelbrot/MandelbrotSetCuda.h"
#include "mandelbrot/MandelbrotSetMPFR.h"

using namespace cv;
using namespace std;

struct ZoomParams {
    double xmin, xmax, ymin, ymax;
};

struct CommandLineArguments {
    size_t width;
    size_t height;
    double x_min, x_max, y_min, y_max;
    double x_center, y_center, xsize;
    bool use_center;
};

CommandLineArguments parseArguments(int argc, char **argv_raw) {
    CommandLineArguments args = {.width = 1024,
                                 .height = 1024,
                                 .x_min = -2.0,
                                 .x_max = 2.0,
                                 .y_min = -2.0,
                                 .y_max = 2.0,
                                 .x_center = 0.0,
                                 .y_center = 0.0,
                                 .xsize = 4.0,
                                 .use_center = false};
    vector<string> argv(argv_raw, argv_raw + argc);
    for (size_t i = 1; i < argc; i++) {
        if (argv[i] == "--resolution") {
            assert(i + 2 < argc);
            args.width = std::stoi(argv[i + 1]);
            args.height = std::stoi(argv[i + 2]);
            i += 2;
        } else if (argv[i] == "--width") {
            assert(i + 1 < argc);
            args.width = std::stoi(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--height") {
            assert(i + 1 < argc);
            args.height = std::stoi(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--xmin") {
            assert(i + 1 < argc);
            args.x_min = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--xmax") {
            assert(i + 1 < argc);
            args.x_max = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--ymin") {
            assert(i + 1 < argc);
            args.y_min = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--ymax") {
            assert(i + 1 < argc);
            args.y_max = std::stod(argv[i + 1]);
            ++i;
        } else if (argv[i] == "--range") {
            assert(i + 4 < argc);
            args.x_min = std::stod(argv[i + 1]);
            args.x_max = std::stod(argv[i + 2]);
            args.y_min = std::stod(argv[i + 3]);
            args.y_max = std::stod(argv[i + 4]);
            i += 4;
        } else if (argv[i] == "--center") {
            assert(i + 3 < argc);
            args.x_center = std::stod(argv[i + 1]);
            args.y_center = std::stod(argv[i + 2]);
            args.xsize = std::stod(argv[i + 3]);
            args.use_center = true;
            i += 3;
        } else if (argv[i] == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --resolution <width> <height>  Set the resolution of the image" << std::endl;
            std::cout << "  --width <width>                Set the width of the image" << std::endl;
            std::cout << "  --height <height>              Set the height of the image" << std::endl;
            std::cout << "  --xmin <xmin>                  Set the minimum x value" << std::endl;
            std::cout << "  --xmax <xmax>                  Set the maximum x value" << std::endl;
            std::cout << "  --ymin <ymin>                  Set the minimum y value" << std::endl;
            std::cout << "  --ymax <ymax>                  Set the maximum y value" << std::endl;
            std::cout << "  --range <xmin> <xmax> <ymin> <ymax> Set the range of x and y values" << std::endl;
            std::cout << "  --center <xcenter> <ycenter> <xsize> Set the center and size of the image" << std::endl;
            std::cout << "  --help                        Display this help message" << std::endl;
            exit(0);
        }
    }

    return args;
}

vector<Mat> precomputeScaleMatrices(const Point2f &center, double scale_rate, int frames) {
    vector<Mat> scale_matrices(frames);
    double factor = 1.0;
    for (int i = 0; i < frames; ++i) {
        scale_matrices[i] = getRotationMatrix2D(center, 0, factor);
        factor *= scale_rate;
    }
    return scale_matrices;
}

#if ENABLE_CUDA
using DefaultMandelbrotSet = Mandelbrot::MandelbrotSetCuda;
constexpr std::string_view CURRENT_IMPLEMENTATION = "CUDA";
#elif ENABLE_MPFR
using DefaultMandelbrotSet = Mandelbrot::MandelbrotSetMPFR;
constexpr std::string_view CURRENT_IMPLEMENTATION = "MPFR";
#else
using DefaultMandelbrotSet = Mandelbrot::MandelbrotSet;
constexpr std::string_view CURRENT_IMPLEMENTATION = "Pure CPU";
#endif

int main(int argc, char **argv) {
    setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    std::cout << "Current implementation: " << CURRENT_IMPLEMENTATION << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Some constants for the zooming animation. We may make them configurable later.
    constexpr int MAX_STEP = 10;
    constexpr double ZOOM_FACTOR = 4.0;
    constexpr double SCALE_RATE = 1.03;


    // Seahorse Valley. We need a more precise center for this.
    constexpr double xcenter = -0.74525, ycenter = 0.12265;

    // Image resolution
    constexpr int width = 4096, height = 4096;

    const int frame_count = ceil(log(ZOOM_FACTOR) / log(SCALE_RATE));

    DefaultMandelbrotSet mandelbrot_set;
    mandelbrot_set.setResolution(width, height);

    VideoWriter writer;
    writer.open("MandelbrotSetCuda.mp4", VideoWriter::fourcc('h', 'v', 'c', 'l'), 30, Size(width, height), true);
    Point2f center(width / 2.0, height / 2.0);


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    auto scale_matrices = precomputeScaleMatrices(center, SCALE_RATE, frame_count);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    cout << "Time taken to precompute scale matrices: " << diff.count() << " seconds" << endl;

    vector<Mat> frames(frame_count);

    for (uint64_t i = 0, factor = (1 << i); i < MAX_STEP; ++i, factor *= ZOOM_FACTOR) {

        mandelbrot_set.setCenter(xcenter, ycenter, 4.0 / factor);

        start = std::chrono::steady_clock::now();
        const auto image = mandelbrot_set.generate();
        end = std::chrono::steady_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Time taken to generate frame " << i + 1 << ": " << diff.count() << " seconds" << std::endl;

        auto filename = std::format("frames/MandelbrotSetCudaFrame{}.png", i + 1);
        imwrite(filename, image);

        if (i == MAX_STEP - 1)
            break;

        std::cout << "Start generating frames between " << i + 1 << " and " << i + 2 << std::endl;

        start = std::chrono::steady_clock::now();

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int idx = 0; idx < frame_count; ++idx) {
            warpAffine(image, frames[idx], scale_matrices[idx], Size(width, height));
        }

        for (auto &frame: frames) {
            writer.write(frame);
        }

        end = std::chrono::steady_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        std::cout << "Time taken to generate frames between " << i + 1 << " and " << i + 2 << ": " << diff.count()
                  << " seconds" << std::endl;
    }

    writer.release();

#if 0
    auto args = parseArguments(argc, argv);

    auto width = args.width;
    auto height = args.height;

    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> diff{};

#if 0
    Mandelbrot::MandelbrotSet mandelbrot(width, height);

    mandelbrot.setXRange(zp.xmin, zp.xmax).setYRange(zp.ymin, zp.ymax);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto image = mandelbrot.generate();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken to generate the image: " << diff.count() << " seconds" << std::endl;

    imwrite("MandelbrotSet.png", image);
#endif


#if ENABLE_CUDA
    Mandelbrot::MandelbrotSetCuda mandelbrot_set_cuda;
    mandelbrot_set_cuda.setResolution(width, height);
    if (args.use_center) {
        mandelbrot_set_cuda.setCenter(args.x_center, args.y_center, args.xsize);
    } else {
        mandelbrot_set_cuda.setXRange(args.x_min, args.x_max).setYRange(args.y_min, args.y_max);
    }

    cout << "Resolution: " << mandelbrot_set_cuda.getWidth() << " x " << mandelbrot_set_cuda.getHeight() << endl;
    cout << fixed << setprecision(2) // NOLINT
         << "XRange: " << mandelbrot_set_cuda.getXMin() << " - " << mandelbrot_set_cuda.getXMax() << endl;
    cout << fixed << setprecision(2) // NOLINT
         << "YRange: " << mandelbrot_set_cuda.getYMin() << " - " << mandelbrot_set_cuda.getYMax() << endl;

    start = std::chrono::steady_clock::now();
    const auto image_cuda = mandelbrot_set_cuda.generate();
    end = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time taken to generate the image using CUDA: " << diff.count() << " seconds" << std::endl;

    imwrite("MandelbrotSetCuda.png", image_cuda);
#endif

#endif

    std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(finish - begin);

    std::cout << "Total time: " << total_time.count() << " seconds" << std::endl;

    return 0;
}
