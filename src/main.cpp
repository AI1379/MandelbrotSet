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
#include <queue>
#include <ranges>
#include <thread>
#include "Algorithm.h"
#include "ColorSchemes.h"
#include "MandelbrotSet.h"
#include "MandelbrotSetCuda.h"
#include "MandelbrotSetMPFR.h"
#include "VideoGenerator.h"

using namespace cv;
using namespace std;
namespace ex = stdexec;

struct CommandLineArguments {
    size_t width;
    size_t height;
    double x_min, x_max, y_min, y_max;
    bool video;
    double x_center, y_center, xsize, ysize;
    int max_step;
    double zoom_factor, scale_rate;
    bool with_key_frames;
    bool set_output;
    string output;
    bool auto_detect, show_grid;
};

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

CommandLineArguments parseArguments(int argc, char **argv_raw) {
    CommandLineArguments args = {
            .width = 2048,
            .height = 2048,
            .x_min = -2.0,
            .x_max = 2.0,
            .y_min = -2.0,
            .y_max = 2.0,
            .video = false,
            .x_center = 0.0,
            .y_center = 0.0,
            .xsize = 4.0,
            .ysize = 5.0,
            .max_step = 10,
            .zoom_factor = 4.0,
            .scale_rate = 1.03,
            .with_key_frames = false,
            .set_output = false,
            .output = "",
            .auto_detect = false,
            .show_grid = false,
    };
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
            assert(i + 4 < argc);
            args.x_center = std::stod(argv[i + 1]);
            args.y_center = std::stod(argv[i + 2]);
            args.xsize = std::stod(argv[i + 3]);
            args.ysize = std::stod(argv[i + 4]);
            i += 4;
        } else if (argv[i] == "--video") {
            assert(i + 3 < argc);
            args.video = true;
            args.max_step = std::stoi(argv[i + 1]);
            args.zoom_factor = std::stod(argv[i + 2]);
            args.scale_rate = std::stod(argv[i + 3]);
            i += 3;
        } else if (argv[i] == "--with-keyframes") {
            args.with_key_frames = true;
        } else if (argv[i] == "-o" || argv[i] == "--output") {
            assert(i + 1 < argc);
            args.set_output = true;
            args.output = argv[i + 1];
            ++i;
        } else if (argv[i] == "--auto-detect") {
            args.auto_detect = true;
        } else if (argv[i] == "--show-grid") {
            args.show_grid = true;
        } else if (argv[i] == "--help") {
            cout << "A simple Mandelbrot set generator" << endl;
            cout << "Current implementation: " << CURRENT_IMPLEMENTATION << endl;
            cout << endl;
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -o / --output <filename>                       Set the filename of the output file" << endl;
            cout << "  --resolution <width> <height>                  Set the resolution of the image" << endl;
            cout << "  --width <width>                                Set the width of the image" << endl;
            cout << "  --height <height>                              Set the height of the image" << endl;
            cout << "  --xmin <xmin>                                  Set the minimum x value" << endl;
            cout << "  --xmax <xmax>                                  Set the maximum x value" << endl;
            cout << "  --ymin <ymin>                                  Set the minimum y value" << endl;
            cout << "  --ymax <ymax>                                  Set the maximum y value" << endl;
            cout << "  --range <xmin> <xmax> <ymin> <ymax>            Set the range of x and y values" << endl;
            cout << "  --center <xcenter> <ycenter> <xsize> <ysize>   Set the center and size for video" << endl;
            cout << "  --video <max_step> <zoom_factor> <scale_rate>  Generate a zooming animation" << endl;
            cout << "  --with-keyframes                               Generate keyframes for the video" << endl;
            cout << "  --auto-detect                                  Automatically detect keyframes" << endl;
            cout << "  --show-grid                                    Show grid on keyframes" << endl;
            cout << "  --help                                         Display this help message" << endl;
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

void generateVideo(const CommandLineArguments &args) {
    // Require arguments: max_step, zoom_factor, scale_rate, x_center, y_center, xsize, width, height
    // Some constants for the zooming animation. We may make them configurable later.
    const int max_step = args.max_step;
    const double zoom_factor = args.zoom_factor;
    const double scale_rate = args.scale_rate;


    // Seahorse Valley. We need a more precise center for this.
    // double xcenter = -0.74525, ycenter = 0.12265;
    const double xcenter = args.x_center, ycenter = args.y_center;

    // Image resolution
    const int width = args.width, height = args.height;

    auto filename = args.set_output ? args.output : "MandelbrotSet.mp4";

    cout << "Generating video..." << endl;
    cout << "Resolution: " << width << " x " << height << endl;
    cout << "Center: " << xcenter << ", " << ycenter << endl;
    cout << "Max step: " << max_step << endl;
    cout << fixed << setprecision(2);
    cout << "Zoom factor: " << zoom_factor << endl;
    cout << "Scale rate: " << scale_rate << endl;

    const int frame_count = ceil(log(zoom_factor) / log(scale_rate));

    DefaultMandelbrotSet mandelbrot_set;
    mandelbrot_set.setResolution(width, height);

    VideoWriter writer;
    writer.open(filename, VideoWriter::fourcc('h', 'v', 'c', 'l'), 30, Size(width, height), true);
    Point2f center(width / 2.0, height / 2.0);


    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto scale_matrices = precomputeScaleMatrices(center, scale_rate, frame_count);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    cout << "Time taken to precompute scale matrices: " << diff.count() << " seconds" << endl;

    vector<Mat> frames(frame_count);

    for (uint64_t i = 0, factor = (1 << i); i < max_step; ++i, factor *= zoom_factor) {

        mandelbrot_set.setCenter(xcenter, ycenter, args.xsize / factor);

        start = std::chrono::steady_clock::now();
        const auto image = mandelbrot_set.generate();
        end = std::chrono::steady_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        cout << "Time taken to generate frame " << i + 1 << ": " << diff.count() << " seconds" << endl;


        if (args.with_key_frames) {
            filename = std::format("frames/MandelbrotSetKeyFrame{}.png", i + 1);
            imwrite(filename, image);
        }

        if (i == max_step - 1)
            break;

        cout << "Start generating frames between " << i + 1 << " and " << i + 2 << endl;

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

        cout << "Time taken to generate frames between " << i + 1 << " and " << i + 2 << ": " << diff.count()
             << " seconds" << endl;
    }

    writer.release();
}

void generateImage(const CommandLineArguments &args) {
    DefaultMandelbrotSet mandelbrot_set;
    mandelbrot_set.setResolution(args.width, args.height);
    mandelbrot_set.setXRange(args.x_min, args.x_max).setYRange(args.y_min, args.y_max);

    cout << fixed << setprecision(2);
    cout << "Resolution: " << mandelbrot_set.getWidth() << " x " << mandelbrot_set.getHeight() << endl;
    cout << "XRange: " << mandelbrot_set.getXMin() << " - " << mandelbrot_set.getXMax() << endl;
    cout << "YRange: " << mandelbrot_set.getYMin() << " - " << mandelbrot_set.getYMax() << endl;

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const auto image_cuda = mandelbrot_set.generate();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    cout << "Time taken to generate the image using CUDA: " << diff.count() << " seconds" << endl;

    auto filename = args.set_output ? args.output : "MandelbrotSetCuda.png";
    imwrite(filename, image_cuda);
}

void asyncGenerateVideo(const CommandLineArguments &args) {
    // Require arguments: max_step, zoom_factor, scale_rate, x_center, y_center, xsize, width, height
    // Some constants for the zooming animation. We may make them configurable later.
    const int max_step = args.max_step;
    const double zoom_factor = args.zoom_factor;
    const double scale_rate = args.scale_rate;


    // Seahorse Valley. We need a more precise center for this.
    // double xcenter = -0.74525, ycenter = 0.12265;
    const double xcenter = args.x_center, ycenter = args.y_center;

    // Image resolution
    const int width = args.width, height = args.height;
    const double xsize = args.xsize, ysize = args.ysize;

    Mandelbrot::VideoGenerator generator;
    generator.setResolution(width, height)
            .setCenter(xcenter, ycenter)
            .setSize(xsize, ysize)
            .setMaxStep(max_step)
            .setZoomFactor(zoom_factor)
            .setScaleRate(scale_rate)
            .setColors(Mandelbrot::randomScheme())
            .setAutoDetect(args.auto_detect)
            .setShowGrid(args.show_grid);

    generator.start();
}

int main(int argc, char **argv) {
    setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    CommandLineArguments args = parseArguments(argc, argv);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cout << "Current implementation: " << CURRENT_IMPLEMENTATION << endl;

#if 1
    if (args.video) {
        asyncGenerateVideo(args);
    } else {
        generateImage(args);
    }
#else

    Mandelbrot::MandelbrotSetCuda mandelbrot_set;
    mandelbrot_set.setResolution(args.width, args.height)
            .setCenter(args.x_center, args.y_center, args.xsize)
            .setColors(Mandelbrot::randomScheme());

    auto esc_map = mandelbrot_set.generateRawMatrix();
    auto image = mandelbrot_set.generate();
    auto high_gradient = Mandelbrot::detectHighGradient(esc_map);

    vector<cv::Rect> bounding_boxes;
    constexpr int DIVIDE = 7;
    constexpr int BLOCK_SIZE = 4;
    auto l = image.cols / DIVIDE * (DIVIDE / 2), r = image.cols / DIVIDE * (DIVIDE / 2 + 1);
    auto t = image.rows / DIVIDE * (DIVIDE / 2), b = image.rows / DIVIDE * (DIVIDE / 2 + 1);
    auto w = image.cols / (DIVIDE * BLOCK_SIZE), h = image.rows / (DIVIDE * BLOCK_SIZE);

#if __cpp_lib_ranges_to_container >= 202202L
    using std::ranges::to;
#else
    using ranges::to;
#endif

    auto sums = views::cartesian_product(views::iota(0, BLOCK_SIZE), views::iota(0, BLOCK_SIZE)) //
                | views::transform([&](const auto &p) {
                      auto [i, j] = p;
                      return std::make_tuple(cv::Rect(l + i * w, t + j * h, w, h), i, j);
                  }) //
                | views::transform([&high_gradient](const auto &p) {
                      const auto &[rect, i, j] = p;
                      return std::make_tuple(cv::sum(high_gradient(rect))[0], i, j);
                  });

    for (auto [sum, x, y]: sums) {
        println(stdout, "Sum: {}, x: {}, y: {}", sum, x, y);
    }

    auto [sum, x, y] = *ranges::max_element(sums, std::less<>(), [](const auto &p) { return std::get<0>(p); });
    auto center = cv::Point2f(l + x * w + w / 2, t + y * h + h / 2);

    Mat high_gradient_image;
    cv::applyColorMap(image, high_gradient_image, cv::COLOR_BGR2GRAY);
    high_gradient_image.setTo(cv::Scalar(0, 0, 255), high_gradient);

    cv::line(image, cv::Point(0, t), cv::Point(image.cols, t), cv::Scalar(0, 255, 0), 2);
    cv::line(image, cv::Point(0, b), cv::Point(image.cols, b), cv::Scalar(0, 255, 0), 2);
    cv::line(image, cv::Point(l, 0), cv::Point(l, image.rows), cv::Scalar(0, 255, 0), 2);
    cv::line(image, cv::Point(r, 0), cv::Point(r, image.rows), cv::Scalar(0, 255, 0), 2);

    for (auto k: views::iota(1, BLOCK_SIZE)) {
        cv::line(image, cv::Point(l + k * w, t), cv::Point(l + k * w, b), cv::Scalar(0, 255, 0), 2);
        cv::line(image, cv::Point(l, t + k * h), cv::Point(r, t + k * h), cv::Scalar(0, 255, 0), 2);
    }

    cv::line(image, cv::Point(0, center.y), cv::Point(image.cols, center.y), cv::Scalar(0, 0, 255), 2);
    cv::line(image, cv::Point(center.x, 0), cv::Point(center.x, image.rows), cv::Scalar(0, 0, 255), 2);
    cv::circle(image, center, 30, cv::Scalar(0, 0, 255), 2);

    imwrite("MandelbrotSetCudaGrad.png", high_gradient_image);
    imwrite("MandelbrotSetCuda.png", image);

#endif

    cout << "Total time: " << TIME_DIFF(begin) << " seconds" << endl;

    return 0;
}
