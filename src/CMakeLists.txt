# It would be better to use local OpenCV installation because it is a large package.
#CPMAddPackage(
#        NAME OpenCV
#        GITHUB_REPOSITORY opencv/opencv
#        GIT_TAG 4.11.0
#        OPTIONS
#        "BUILD_SHARED_LIBS OFF" "BUILD_TESTS OFF" "BUILD_PERF_TESTS OFF" "BUILD_EXAMPLES OFF" "BUILD_LIST imgproc,core,videoio"
#)
# Unfortunately OpenCV does not support add_directory() so we have to use find_package().
find_package(OpenCV REQUIRED)

CPMAddPackage(
        NAME fmt
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.1.4
)

set(MANDELBROT_SET_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ColorSchemes.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Algorithm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

set(MANDELBROT_SET_DEPENDENCIES
        opencv_core
        opencv_imgproc
        opencv_videoio
        fmt::fmt
)

if (ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        add_definitions(-DENABLE_OPENMP)
        list(APPEND MANDELBROT_SET_DEPENDENCIES OpenMP::OpenMP_CXX)
    elseif (AUTO_DISABLE_REQ)
        message(STATUS "OpenMP is required but not supported by the compiler")
        set(ENABLE_OPENMP OFF)
    else ()
        message(FATAL_ERROR "OpenMP is required but not supported by the compiler")
    endif ()
endif ()

if (ENABLE_CUDA)
    list(APPEND MANDELBROT_SET_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSetCuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/ExtendedDouble.cu)
    add_definitions(-DENABLE_CUDA)
endif ()

if (ENABLE_STDEXEC)
    CPMAddPackage(
            NAME STDEXEC
            GITHUB_REPOSITORY NVIDIA/stdexec
            GIT_TAG main
    )
    add_definitions(-DENABLE_STDEXEC)
    list(APPEND MANDELBROT_SET_DEPENDENCIES STDEXEC::stdexec)
endif ()

message(STATUS "Project sources: ${MANDELBROT_SET_SOURCE}")
message(STATUS "Project dependencies: ${MANDELBROT_SET_DEPENDENCIES}")

include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(MandelbrotSet ${MANDELBROT_SET_SOURCE})

if (ENABLE_CUDA)
    # For Cuda. From https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt
    target_compile_features(MandelbrotSet PUBLIC cxx_std_23)
    set_target_properties(MandelbrotSet
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
endif ()

target_link_libraries(MandelbrotSet PRIVATE ${MANDELBROT_SET_DEPENDENCIES})