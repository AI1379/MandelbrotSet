if (NOT TARGET STDEXEC::stdexec)
    if (STDEXEC_ROOT AND NOT STDEXEC_ROOT)
        set(STDEXEC_ROOT
                ${STDEXEC_ROOT}
                CACHE PATH "stdexec base directory"
        )
        unset(STDEXEC_ROOT CACHE)
    endif ()

    if (STDEXEC_DIR AND NOT STDEXEC_DIR)
        set(STDEXEC_ROOT
                ${STDEXEC_DIR}
                CACHE PATH "stdexec base directory"
        )
        unset(STDEXEC_DIR CACHE)
    endif ()

    find_path(STDEXEC_SOURCE_DIR include/stdexec/execution.hpp
            HINTS ${STDEXEC_ROOT}
            PATHS
            ENV STDEXEC_ROOT
            ENV STDEXEC_DIR
    )
    set(STDEXEC_INCLUDE_DIR ${STDEXEC_SOURCE_DIR}/include)
    message(STATUS "stdexec include dir: ${STDEXEC_INCLUDE_DIR}")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(
            STDEXEC
            REQUIRED_VARS STDEXEC_SOURCE_DIR
            FAIL_MESSAGE "stdexec not found"
    )
    if (STDEXEC_FOUND)
        file(TO_CMAKE_PATH ${STDEXEC_INCLUDE_DIR} STDEXEC_INCLUDE_DIR)
        add_library(STDEXEC::stdexec INTERFACE IMPORTED)
        target_include_directories(
                STDEXEC::stdexec SYSTEM INTERFACE ${STDEXEC_INCLUDE_DIR}
        )

        mark_as_advanced(STDEXEC_INCLUDE_DIR STDEXEC_ROOT)
    endif ()

endif ()