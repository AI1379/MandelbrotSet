//
// Created by Renatus Madrigal on 3/21/2025.
//

#ifndef MANDELBROTSET_SRC_MANDELBROTSET_COLORSCHEMES_H
#define MANDELBROTSET_SRC_MANDELBROTSET_COLORSCHEMES_H

/**
 * @file ColorSchemes.h
 */

#include "BaseMandelbrotSet.h"

namespace Mandelbrot {

    /**
     * @brief Color scheme 1.
     * @return A pointer to the color scheme array.
     */
    ColorSchemeType colorScheme1();

    /**
     * @brief Color scheme 2.
     * @return A pointer to the color scheme array.
     */
    ColorSchemeType colorScheme2();

    /**
     * @brief Random color scheme.
     * @return A pointer to the color scheme array.
     * @note The color scheme is randomly generated. So, it is not deterministic.
     */
    ColorSchemeType randomScheme();

    /**
     * @brief Normal distribution color scheme.
     * @return A pointer to the color scheme array.
     * @note The color scheme is generated using normal distribution.
     */
    ColorSchemeType normalDistScheme();

} // namespace Mandelbrot

#endif // MANDELBROTSET_SRC_MANDELBROTSET_COLORSCHEMES_H
