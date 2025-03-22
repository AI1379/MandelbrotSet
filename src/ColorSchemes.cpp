//
// Created by Renatus Madrigal on 3/21/2025.
//

#include "ColorSchemes.h"
#include <numbers>
#include <random>

namespace Mandelbrot {

    ColorSchemeType colorScheme1() {
        static cv::Vec3b colors[MAX_ITERATIONS + 1] = {};
        static bool initialized = false;

        if (initialized)
            return colors;

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
        initialized = true;

        return colors;
    }

    ColorSchemeType colorScheme2() {
        static cv::Vec3b colors[MAX_ITERATIONS + 1];
        static bool initialized = false;

        if (initialized)
            return colors;

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
        initialized = true;

        return colors;
    }

    ColorSchemeType randomScheme() {
        static std::random_device rd;
        static std::mt19937 rng(rd());
        static cv::Vec3b colors[MAX_ITERATIONS + 1];
        std::uniform_real_distribution<double> mean_dist(64, 192);
        std::normal_distribution<double> dist(0, 1.0);
        auto mean_r = mean_dist(rng), mean_g = mean_dist(rng), mean_b = mean_dist(rng);

        auto gen = [&](unsigned mean) -> uint8_t {
            auto seed = dist(rng);
            auto pct = std::atan(seed) * std::numbers::inv_pi_v<double> * 2;
            auto l = mean, r = 255 - mean;
            if (pct < 0)
                return static_cast<uint8_t>(l * pct + mean);
            else
                return static_cast<uint8_t>(r * pct + mean);
        };
        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            colors[n] = {gen(mean_r), gen(mean_g), gen(mean_b)};
        }
        colors[MAX_ITERATIONS] = {0, 0, 0};

        return colors;
    }

    ColorSchemeType normalDistScheme() {
        static std::random_device rd;
        static std::mt19937 rng(rd());
        static cv::Vec3b colors[MAX_ITERATIONS + 1];
        std::normal_distribution<double> dist(128, 64);
        auto gen = [&](double mean) -> uint8_t {
            auto seed = dist(rng);
            return static_cast<uint8_t>(std::clamp(seed, 0.0, 255.0));
        };
        for (int n = 0; n < MAX_ITERATIONS; ++n) {
            colors[n] = {gen(128), gen(128), gen(128)};
        }
        colors[MAX_ITERATIONS] = {0, 0, 0};

        return colors;
    }

} // namespace Mandelbrot
