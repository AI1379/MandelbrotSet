find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OPENMP)
endif ()

set(MANDELBROT_SET_LIBRARY_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

if (ENABLE_MPFR)
    find_package(MPFR REQUIRED)
    list(APPEND MANDELBROT_SET_LIBRARY_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSetMPFR.cpp)
    message(STATUS "MPFR include dir: ${MPFR_INCLUDE_DIR}")
    add_definitions(-DENABLE_MPFR)
endif ()

if (ENABLE_CUDA)
    list(APPEND MANDELBROT_SET_LIBRARY_SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/MandelbrotSetCuda.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/ExtendedDouble.cu)
    add_definitions(-DENABLE_CUDA)
endif ()

message(STATUS "Library sources: ${MANDELBROT_SET_LIBRARY_SOURCE}")

include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(MandelbrotSet ${MANDELBROT_SET_LIBRARY_SOURCE})

if (ENABLE_CUDA)
    # For Cuda. From https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt
    target_compile_features(MandelbrotSet PUBLIC cxx_std_23)
    set_target_properties(MandelbrotSet
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
endif ()

target_link_libraries(MandelbrotSet PUBLIC ${OpenCV_LIBS})

if (ENABLE_MPFR)
    target_link_libraries(MandelbrotSet PUBLIC MPFR::MPFR)
endif ()

if (ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(MandelbrotSet PUBLIC OpenMP::OpenMP_CXX)
endif ()
