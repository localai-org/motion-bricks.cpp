cmake_minimum_required(VERSION 3.25)
project(motionbricks VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

if(WIN32)
    # Executables and dependent DLLs must share the build-tree directory.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Keep installed libraries and backend modules relocatable together.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

option(MOTIONBRICKS_BUILD_TESTS "Build motion-bricks.cpp tests" ON)
option(MOTIONBRICKS_ENABLE_GGML "Use the pinned GGML submodule when present" ON)
option(MOTIONBRICKS_CPU_ALL_VARIANTS "Build GGML CPU variants and select at runtime" ON)
option(MOTIONBRICKS_ENABLE_VULKAN "Enable the GGML Vulkan backend" ON)
option(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS "Run hardware-dependent Vulkan parity tests" OFF)
option(MOTIONBRICKS_ENABLE_FUZZERS "Build libFuzzer targets" OFF)
option(MOTIONBRICKS_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MOTIONBRICKS_ENABLE_PHYSICS "Enable native MuJoCo / GGML SONIC simulation (3.12.0 or 3.3.5 reference)" OFF)
option(MOTIONBRICKS_DOWNLOAD_MODELS "Download and verify the published runtime model/style bundle" ON)
set(MOTIONBRICKS_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH
    "Pinned GGML submodule source directory")
set(MOTIONBRICKS_MODEL_REPOSITORY "LocalAI-io/MotionBricks-G1-GGML" CACHE STRING
    "Hugging Face repository containing the runtime bundle")
set(MOTIONBRICKS_MODEL_REVISION "cc2a47603dbc203a4f18f35dd06ed3611833f506" CACHE STRING
    "Pinned Hugging Face revision containing the runtime bundle")
set(MOTIONBRICKS_REFERENCE_BUNDLE "${CMAKE_CURRENT_SOURCE_DIR}/generated/g1-f32" CACHE PATH
    "Downloaded or locally converted MotionBricks runtime bundle")
set(MOTIONBRICKS_REFERENCE_STYLES "${CMAKE_CURRENT_SOURCE_DIR}/generated/styles" CACHE PATH
    "Downloaded or locally converted MotionBricks style bundle")
set(MOTIONBRICKS_VULKAN_PARITY_DEVICE "0" CACHE STRING
    "GGML Vulkan device index used by opt-in parity tests")
set(MOTIONBRICKS_VULKAN_PARITY_HARDWARE "unspecified" CACHE STRING
    "Hardware name recorded by opt-in Vulkan parity tests")
set(MOTIONBRICKS_VULKAN_PARITY_DRIVER "unspecified" CACHE STRING
    "Driver version recorded by opt-in Vulkan parity tests")
if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS AND
   (MOTIONBRICKS_VULKAN_PARITY_HARDWARE STREQUAL "unspecified" OR
    MOTIONBRICKS_VULKAN_PARITY_DRIVER STREQUAL "unspecified"))
    message(FATAL_ERROR
        "Vulkan parity tests require explicit MOTIONBRICKS_VULKAN_PARITY_HARDWARE "
        "and MOTIONBRICKS_VULKAN_PARITY_DRIVER tags")
endif()

if(MOTIONBRICKS_ENABLE_GGML AND MOTIONBRICKS_DOWNLOAD_MODELS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_gguf_weights.py"
            --output "${CMAKE_CURRENT_SOURCE_DIR}/generated"
            --repo "${MOTIONBRICKS_MODEL_REPOSITORY}"
            --revision "${MOTIONBRICKS_MODEL_REVISION}"
        RESULT_VARIABLE MOTIONBRICKS_MODEL_DOWNLOAD_RESULT
        COMMAND_ECHO STDOUT)
    if(NOT MOTIONBRICKS_MODEL_DOWNLOAD_RESULT EQUAL 0)
        message(FATAL_ERROR
            "MotionBricks model download/verification failed. Set "
            "MOTIONBRICKS_DOWNLOAD_MODELS=OFF for a source-only build.")
    endif()
endif()

set(MOTIONBRICKS_SOURCES
    src/capi.cpp
    src/inference.cpp
    src/decoder.cpp
    src/error.cpp
    src/model.cpp
    src/neural_runtime.cpp
    src/pose.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/root.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/motion_rep.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/planner.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/sampling.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/style.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/agent.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/sonic.cpp)
list(APPEND MOTIONBRICKS_SOURCES src/sonic_physics.cpp)

add_library(motionbricks_shared SHARED ${MOTIONBRICKS_SOURCES})
add_library(motionbricks::motionbricks ALIAS motionbricks_shared)
set_target_properties(motionbricks_shared PROPERTIES
    OUTPUT_NAME motionbricks
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    EXPORT_NAME motionbricks
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0)
target_compile_definitions(motionbricks_shared PRIVATE MOTIONBRICKS_BUILD MOTIONBRICKS_SHARED)

add_library(motionbricks_static STATIC ${MOTIONBRICKS_SOURCES})
add_library(motionbricks::motionbricks_static ALIAS motionbricks_static)
if(WIN32)
    # Import libraries and static libraries share the .lib extension, so the
    # shared and static targets cannot also share an output name.
    set(MOTIONBRICKS_STATIC_OUTPUT_NAME motionbricks_static)
else()
    set(MOTIONBRICKS_STATIC_OUTPUT_NAME motionbricks)
endif()
set_target_properties(motionbricks_static PROPERTIES
    OUTPUT_NAME ${MOTIONBRICKS_STATIC_OUTPUT_NAME}
    EXPORT_NAME motionbricks_static)
target_compile_definitions(motionbricks_static PRIVATE MOTIONBRICKS_BUILD)

foreach(target motionbricks_shared motionbricks_static)
    target_include_directories(${target}
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endforeach()

if(MOTIONBRICKS_ENABLE_GGML)
    if(EXISTS "${MOTIONBRICKS_GGML_SOURCE_DIR}/CMakeLists.txt")
        set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
        set(GGML_VULKAN ${MOTIONBRICKS_ENABLE_VULKAN} CACHE BOOL "" FORCE)
        set(GGML_BACKEND_DL ${MOTIONBRICKS_CPU_ALL_VARIANTS} CACHE BOOL "" FORCE)
        set(GGML_CPU_ALL_VARIANTS ${MOTIONBRICKS_CPU_ALL_VARIANTS} CACHE BOOL "" FORCE)
        if(MOTIONBRICKS_CPU_ALL_VARIANTS)
            set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
        endif()
        # GGML's module targets require an output directory, including when
        # only a library target (rather than an executable) is requested.
        block()
            set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
            add_subdirectory("${MOTIONBRICKS_GGML_SOURCE_DIR}"
                             "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        endblock()
        get_target_property(MOTIONBRICKS_GGML_HEADERS ggml PUBLIC_HEADER)
        list(TRANSFORM MOTIONBRICKS_GGML_HEADERS PREPEND "${MOTIONBRICKS_GGML_SOURCE_DIR}/")
        set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${MOTIONBRICKS_GGML_HEADERS}")
        # Install explicitly: the vendored subdirectory is EXCLUDE_FROM_ALL.
        # Modules live beside libggml so discovery also works for FFI callers.
        set(MOTIONBRICKS_GGML_TARGETS ggml ggml-base ${GGML_AVAILABLE_BACKENDS})
        set(MOTIONBRICKS_BACKEND_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
        set(MOTIONBRICKS_BACKEND_INSTALL_DIRECTORY "${CMAKE_INSTALL_LIBDIR}")
        if(WIN32)
            set(MOTIONBRICKS_BACKEND_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
            set(MOTIONBRICKS_BACKEND_INSTALL_DIRECTORY "${CMAKE_INSTALL_BINDIR}")
        endif()
        foreach(backend IN LISTS MOTIONBRICKS_GGML_TARGETS)
            if(TARGET ${backend})
                set_target_properties(${backend} PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY "${MOTIONBRICKS_BACKEND_OUTPUT_DIRECTORY}"
                    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
                install(TARGETS ${backend}
                    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                    LIBRARY DESTINATION ${MOTIONBRICKS_BACKEND_INSTALL_DIRECTORY}
                    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
            endif()
        endforeach()
        target_link_libraries(motionbricks_shared PRIVATE "$<BUILD_INTERFACE:ggml>")
        target_link_libraries(motionbricks_static PRIVATE "$<BUILD_INTERFACE:ggml>")
        target_link_libraries(motionbricks_shared PRIVATE ${CMAKE_DL_LIBS})
        target_link_libraries(motionbricks_static PRIVATE ${CMAKE_DL_LIBS})
        target_compile_definitions(motionbricks_shared PRIVATE MOTIONBRICKS_HAVE_GGML=1)
        target_compile_definitions(motionbricks_static PRIVATE MOTIONBRICKS_HAVE_GGML=1)
        if(MOTIONBRICKS_ENABLE_VULKAN AND TARGET ggml-vulkan)
            target_compile_definitions(motionbricks_shared PRIVATE MOTIONBRICKS_HAVE_VULKAN=1)
            target_compile_definitions(motionbricks_static PRIVATE MOTIONBRICKS_HAVE_VULKAN=1)
        endif()
    else()
        message(STATUS "GGML submodule not present; building the ABI scaffold only")
    endif()
endif()

if(MOTIONBRICKS_SANITIZE)
    if(MSVC)
        message(FATAL_ERROR "MOTIONBRICKS_SANITIZE currently requires Clang or GCC")
    endif()
    foreach(target motionbricks_shared motionbricks_static)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()

if(MOTIONBRICKS_ENABLE_PHYSICS)
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h REQUIRED)
    find_library(MUJOCO_LIBRARY NAMES mujoco libmujoco.so.3.12.0 libmujoco.so.3.3.5 REQUIRED)
    foreach(target motionbricks_shared motionbricks_static)
        target_include_directories(${target} PRIVATE "${MUJOCO_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE "${MUJOCO_LIBRARY}")
        target_compile_definitions(${target} PRIVATE MOTIONBRICKS_HAVE_MUJOCO=1)
    endforeach()
    if(MOTIONBRICKS_ENABLE_GGML AND UNIX AND NOT APPLE)
        add_library(motionbricks-sonic-profile SHARED EXCLUDE_FROM_ALL reference/sonic_profile.cpp)
        target_include_directories(motionbricks-sonic-profile PRIVATE
            "${MUJOCO_INCLUDE_DIR}" "${MOTIONBRICKS_GGML_SOURCE_DIR}/include")
        target_link_options(motionbricks-sonic-profile PRIVATE "-Wl,--no-as-needed")
        target_link_libraries(motionbricks-sonic-profile PRIVATE motionbricks_shared ${CMAKE_DL_LIBS})
    endif()
endif()

add_executable(motionbricks-cli src/cli.cpp src/session_replay.cpp)
target_link_libraries(motionbricks-cli PRIVATE motionbricks::motionbricks)
set_target_properties(motionbricks-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(motionbricks-parity src/parity_runner.cpp src/plan_parity.cpp)
target_include_directories(motionbricks-parity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(motionbricks-parity PRIVATE motionbricks::motionbricks_static)
set_target_properties(motionbricks-parity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(MOTIONBRICKS_ENABLE_GGML)
    add_executable(motionbricks-cpu-profile EXCLUDE_FROM_ALL reference/motionbricks_cpu_profile.cpp)
    target_include_directories(motionbricks-cpu-profile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(motionbricks-cpu-profile PRIVATE motionbricks::motionbricks_static)
    set_target_properties(motionbricks-cpu-profile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_executable(motionbricks-batch-profile EXCLUDE_FROM_ALL reference/motionbricks_batch_profile.cpp)
    target_include_directories(motionbricks-batch-profile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(motionbricks-batch-profile PRIVATE motionbricks::motionbricks_static)
    set_target_properties(motionbricks-batch-profile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

include(CTest)
if(TARGET ggml AND NOT GGML_BACKEND_DL AND GGML_AVX512 AND UNIX AND NOT APPLE)
    # Explicit opt-in diagnostic target; never installed or linked into runtime.
    add_library(motionbricks-sonic-cpu-experiment SHARED EXCLUDE_FROM_ALL
        reference/sonic_cpu_experiment.cpp)
    target_link_libraries(motionbricks-sonic-cpu-experiment PRIVATE ggml ${CMAKE_DL_LIBS})
    target_compile_options(motionbricks-sonic-cpu-experiment PRIVATE
        -mavx512f -mavx512dq -mavx512vl -mf16c -mfma)
endif()
add_executable(motionbricks-sonic-parity src/sonic_parity.cpp)
target_link_libraries(motionbricks-sonic-parity PRIVATE motionbricks::motionbricks)
set_target_properties(motionbricks-sonic-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(MOTIONBRICKS_SANITIZE)
    target_compile_options(motionbricks-sonic-parity PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(motionbricks-sonic-parity PRIVATE -fsanitize=address,undefined)
endif()
if(BUILD_TESTING AND MOTIONBRICKS_BUILD_TESTS)
    add_subdirectory(tests)
endif()

install(TARGETS motionbricks_shared motionbricks_static motionbricks-cli motionbricks-parity
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
