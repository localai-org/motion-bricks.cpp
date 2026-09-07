add_executable(motionbricks-c-api-test c_api_test.c)
target_link_libraries(motionbricks-c-api-test PRIVATE motionbricks::motionbricks)
add_test(NAME motionbricks-c-api COMMAND motionbricks-c-api-test)

add_executable(motionbricks-inference-api-test inference_api_test.c)
target_link_libraries(motionbricks-inference-api-test PRIVATE motionbricks::motionbricks)
add_test(NAME motionbricks-inference-api COMMAND motionbricks-inference-api-test)
add_library(motionbricks-inference-example OBJECT "${PROJECT_SOURCE_DIR}/examples/inference.c")
target_include_directories(motionbricks-inference-example PRIVATE "${PROJECT_SOURCE_DIR}/include")
if(MOTIONBRICKS_ENABLE_GGML AND EXISTS "${MOTIONBRICKS_REFERENCE_BUNDLE}/pose.gguf" AND EXISTS "${MOTIONBRICKS_REFERENCE_STYLES}/walk.mbstyle")
    add_executable(motionbricks-inference-model-test inference_model_test.cpp)
    target_sources(motionbricks-inference-model-test PRIVATE $<TARGET_OBJECTS:motionbricks-inference-example>)
    target_include_directories(motionbricks-inference-model-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(motionbricks-inference-model-test PRIVATE motionbricks_static)
    add_test(NAME motionbricks-inference-model COMMAND motionbricks-inference-model-test
        "${MOTIONBRICKS_REFERENCE_BUNDLE}" "${MOTIONBRICKS_REFERENCE_STYLES}/walk.mbstyle")
    if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        add_test(NAME motionbricks-inference-model-vulkan COMMAND motionbricks-inference-model-test
            "${MOTIONBRICKS_REFERENCE_BUNDLE}" "${MOTIONBRICKS_REFERENCE_STYLES}/walk.mbstyle" vulkan)
        set_tests_properties(motionbricks-inference-model-vulkan PROPERTIES ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
    endif()
endif()

add_executable(motionbricks-cpp-wrapper-test cpp_wrapper_test.cpp)
target_link_libraries(motionbricks-cpp-wrapper-test PRIVATE motionbricks::motionbricks)
add_test(NAME motionbricks-cpp-wrapper COMMAND motionbricks-cpp-wrapper-test)

add_executable(motionbricks-session-replay-test
    session_replay_test.cpp
    "${PROJECT_SOURCE_DIR}/src/session_replay.cpp")
target_include_directories(motionbricks-session-replay-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
add_test(NAME motionbricks-session-replay COMMAND motionbricks-session-replay-test)

add_executable(motionbricks-plan-parity-format-test
    plan_parity_format_test.cpp
    "${PROJECT_SOURCE_DIR}/src/plan_parity.cpp")
target_include_directories(motionbricks-plan-parity-format-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
add_test(NAME motionbricks-plan-parity-format COMMAND motionbricks-plan-parity-format-test)

add_executable(motionbricks-context-endpoints-test context_endpoints_test.cpp)
target_include_directories(motionbricks-context-endpoints-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(motionbricks-context-endpoints-test PRIVATE motionbricks_static)
add_test(NAME motionbricks-context-endpoints COMMAND motionbricks-context-endpoints-test)

add_executable(motionbricks-sampling-test sampling_test.cpp "${PROJECT_SOURCE_DIR}/src/sampling.cpp")
target_include_directories(motionbricks-sampling-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
add_test(NAME motionbricks-sampling COMMAND motionbricks-sampling-test)
add_test(NAME motionbricks-sampling-upstream COMMAND motionbricks-sampling-test
    "${PROJECT_SOURCE_DIR}/tests/fixtures/upstream-gumbel.mbsampling")

add_executable(motionbricks-agent-blend-test agent_blend_test.cpp)
target_include_directories(motionbricks-agent-blend-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(motionbricks-agent-blend-test PRIVATE motionbricks_static)
if(EXISTS "${PROJECT_SOURCE_DIR}/generated/context-blend.mbblend")
    add_test(NAME motionbricks-agent-blend COMMAND motionbricks-agent-blend-test
        "${PROJECT_SOURCE_DIR}/generated/context-blend.mbblend")
else()
    add_test(NAME motionbricks-agent-blend COMMAND motionbricks-agent-blend-test)
endif()

set(MOTIONBRICKS_REFERENCE_FIXTURES "${PROJECT_SOURCE_DIR}/generated/fixtures-gguf" CACHE PATH
    "Optional local GGUF reference fixtures for parity tests")
if(MOTIONBRICKS_ENABLE_GGML AND
   EXISTS "${MOTIONBRICKS_REFERENCE_BUNDLE}/pose.gguf" AND
   EXISTS "${MOTIONBRICKS_REFERENCE_FIXTURES}/pose.gguf")
    add_executable(motionbricks-pose-parity-test pose_parity_test.cpp)
    target_include_directories(motionbricks-pose-parity-test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${MOTIONBRICKS_GGML_SOURCE_DIR}/include")
    target_compile_definitions(motionbricks-pose-parity-test PRIVATE
        MOTIONBRICKS_HAVE_GGML=1
        MOTIONBRICKS_REFERENCE_BUNDLE="${MOTIONBRICKS_REFERENCE_BUNDLE}"
        MOTIONBRICKS_REFERENCE_FIXTURES="${MOTIONBRICKS_REFERENCE_FIXTURES}")
    target_link_libraries(motionbricks-pose-parity-test PRIVATE motionbricks_static ggml)
    add_test(NAME motionbricks-pose-parity COMMAND motionbricks-pose-parity-test)

    add_executable(motionbricks-decoder-parity-test decoder_parity_test.cpp)
    target_include_directories(motionbricks-decoder-parity-test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${MOTIONBRICKS_GGML_SOURCE_DIR}/include")
    target_compile_definitions(motionbricks-decoder-parity-test PRIVATE
        MOTIONBRICKS_HAVE_GGML=1
        MOTIONBRICKS_REFERENCE_BUNDLE="${MOTIONBRICKS_REFERENCE_BUNDLE}"
        MOTIONBRICKS_REFERENCE_FIXTURES="${MOTIONBRICKS_REFERENCE_FIXTURES}")
    target_link_libraries(motionbricks-decoder-parity-test PRIVATE motionbricks_static ggml)
    add_test(NAME motionbricks-decoder-parity COMMAND motionbricks-decoder-parity-test)

    add_executable(motionbricks-root-parity-test root_parity_test.cpp)
    target_include_directories(motionbricks-root-parity-test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${MOTIONBRICKS_GGML_SOURCE_DIR}/include")
    target_compile_definitions(motionbricks-root-parity-test PRIVATE
        MOTIONBRICKS_HAVE_GGML=1
        MOTIONBRICKS_REFERENCE_BUNDLE="${MOTIONBRICKS_REFERENCE_BUNDLE}"
        MOTIONBRICKS_REFERENCE_FIXTURES="${MOTIONBRICKS_REFERENCE_FIXTURES}")
    target_link_libraries(motionbricks-root-parity-test PRIVATE motionbricks_static ggml)
    add_test(NAME motionbricks-root-parity COMMAND motionbricks-root-parity-test)

    if(EXISTS "${MOTIONBRICKS_REFERENCE_STYLES}/walk.mbstyle")
        add_executable(motionbricks-agent-integration-test agent_integration_test.cpp)
        target_compile_definitions(motionbricks-agent-integration-test PRIVATE
            MOTIONBRICKS_REFERENCE_BUNDLE="${MOTIONBRICKS_REFERENCE_BUNDLE}"
            MOTIONBRICKS_REFERENCE_STYLES="${MOTIONBRICKS_REFERENCE_STYLES}")
        target_link_libraries(motionbricks-agent-integration-test PRIVATE motionbricks_static)
        add_test(NAME motionbricks-agent-integration COMMAND motionbricks-agent-integration-test)

        add_executable(motionbricks-agent-backend-parity-test agent_backend_parity_test.cpp)
        target_compile_definitions(motionbricks-agent-backend-parity-test PRIVATE
            MOTIONBRICKS_REFERENCE_BUNDLE="${MOTIONBRICKS_REFERENCE_BUNDLE}"
            MOTIONBRICKS_REFERENCE_STYLES="${MOTIONBRICKS_REFERENCE_STYLES}")
        target_link_libraries(motionbricks-agent-backend-parity-test PRIVATE motionbricks_static)

        if(EXISTS "${PROJECT_SOURCE_DIR}/generated/open-loop-parity/open-loop.mbparity")
            set(MOTIONBRICKS_CPU_PARITY_TRACE_ARGS)
            if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
                list(APPEND MOTIONBRICKS_CPU_PARITY_TRACE_ARGS
                    --trace-directory "${PROJECT_BINARY_DIR}/open-loop-cpu-traces")
            endif()
            add_test(NAME motionbricks-open-loop-parity-cpu
                COMMAND motionbricks-parity
                    "${PROJECT_SOURCE_DIR}/generated/open-loop-parity/open-loop.mbparity"
                    "${MOTIONBRICKS_REFERENCE_BUNDLE}"
                    "${MOTIONBRICKS_REFERENCE_STYLES}"
                    "${PROJECT_BINARY_DIR}/open-loop-cpu-report.json" cpu --report-only
                    ${MOTIONBRICKS_CPU_PARITY_TRACE_ARGS})
            set_tests_properties(motionbricks-open-loop-parity-cpu PROPERTIES TIMEOUT 600)
        endif()
    endif()
endif()

find_program(MOTIONBRICKS_GO_EXECUTABLE go)
if(MOTIONBRICKS_GO_EXECUTABLE AND NOT MOTIONBRICKS_SANITIZE)
    add_test(NAME motionbricks-purego-abi
        COMMAND "${CMAKE_COMMAND}" -E env
            "CGO_ENABLED=0"
            "MOTIONBRICKS_LIB=$<TARGET_FILE:motionbricks_shared>"
            "${MOTIONBRICKS_GO_EXECUTABLE}" test ./...)
    set_tests_properties(motionbricks-purego-abi PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bindings/go")

    find_program(MOTIONBRICKS_CHROMIUM_EXECUTABLE NAMES chromium chromium-browser google-chrome)
    if(MOTIONBRICKS_CHROMIUM_EXECUTABLE AND
       MOTIONBRICKS_ENABLE_GGML AND
       EXISTS "${MOTIONBRICKS_REFERENCE_BUNDLE}/pose.gguf" AND
       EXISTS "${MOTIONBRICKS_REFERENCE_STYLES}/walk.mbstyle")
        set(MOTIONBRICKS_DEMO_TEST_ENV
            "CGO_ENABLED=0"
            "MOTIONBRICKS_LIB=$<TARGET_FILE:motionbricks_shared>"
            "MOTIONBRICKS_MODEL=${MOTIONBRICKS_REFERENCE_BUNDLE}"
            "MOTIONBRICKS_STYLES=${MOTIONBRICKS_REFERENCE_STYLES}"
            "MOTIONBRICKS_CHROME=${MOTIONBRICKS_CHROMIUM_EXECUTABLE}")
        if(EXISTS "${PROJECT_SOURCE_DIR}/generated/session-replay/session.mbreplay")
            list(APPEND MOTIONBRICKS_DEMO_TEST_ENV
                "MOTIONBRICKS_REPLAY=${PROJECT_SOURCE_DIR}/generated/session-replay/session.mbreplay")
        endif()
        if(EXISTS "${PROJECT_SOURCE_DIR}/generated/open-loop-parity/open-loop.mbparity")
            list(APPEND MOTIONBRICKS_DEMO_TEST_ENV
                "MOTIONBRICKS_COMPARISON=${PROJECT_BINARY_DIR}/open-loop-cpu-report.json"
                "MOTIONBRICKS_COMPARISON_SCREENSHOT=${PROJECT_BINARY_DIR}/threejs-open-loop-comparison.png")
        endif()
        add_test(NAME motionbricks-go-demo
            COMMAND "${CMAKE_COMMAND}" -E env
                ${MOTIONBRICKS_DEMO_TEST_ENV}
                "${MOTIONBRICKS_GO_EXECUTABLE}" test -v ./...)
        set_tests_properties(motionbricks-go-demo PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../demo"
            RUN_SERIAL TRUE
            TIMEOUT 60)
        if(TEST motionbricks-open-loop-parity-cpu)
            set_tests_properties(motionbricks-go-demo PROPERTIES
                DEPENDS motionbricks-open-loop-parity-cpu)
        endif()
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import numpy, safetensors"
        RESULT_VARIABLE MOTIONBRICKS_SESSION_TEST_DEPS)
    if(MOTIONBRICKS_SESSION_TEST_DEPS EQUAL 0)
        add_test(NAME motionbricks-reference-session-tools
            COMMAND "${Python3_EXECUTABLE}" -m unittest reference/test_session_tools.py)
        set_tests_properties(motionbricks-reference-session-tools PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif()
endif()

if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS AND
   MOTIONBRICKS_ENABLE_VULKAN AND
   Python3_Interpreter_FOUND AND
   TEST motionbricks-open-loop-parity-cpu)
    foreach(component root pose decoder)
        add_test(NAME motionbricks-${component}-parity-vulkan
            COMMAND motionbricks-${component}-parity-test vulkan)
        set_tests_properties(motionbricks-${component}-parity-vulkan PROPERTIES
            ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}"
            TIMEOUT 120)
    endforeach()
    add_test(NAME motionbricks-agent-backend-parity-vulkan
        COMMAND motionbricks-agent-backend-parity-test)
    set_tests_properties(motionbricks-agent-backend-parity-vulkan PROPERTIES
        ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}"
        TIMEOUT 120)

    add_test(NAME motionbricks-open-loop-parity-vulkan
        COMMAND motionbricks-parity
            "${PROJECT_SOURCE_DIR}/generated/open-loop-parity/open-loop.mbparity"
            "${MOTIONBRICKS_REFERENCE_BUNDLE}"
            "${MOTIONBRICKS_REFERENCE_STYLES}"
            "${PROJECT_BINARY_DIR}/open-loop-vulkan-report.json" vulkan
            --trace-directory "${PROJECT_BINARY_DIR}/open-loop-vulkan-traces")
    set_tests_properties(motionbricks-open-loop-parity-vulkan PROPERTIES
        ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}"
        TIMEOUT 600)

    add_test(NAME motionbricks-open-loop-backend-parity
        COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/reference/compare_plan_reports.py"
            "${PROJECT_BINARY_DIR}/open-loop-cpu-report.json"
            "${PROJECT_BINARY_DIR}/open-loop-vulkan-report.json"
            --output "${PROJECT_BINARY_DIR}/open-loop-cpu-vulkan-comparison.json"
            --hardware "${MOTIONBRICKS_VULKAN_PARITY_HARDWARE}"
            --driver "${MOTIONBRICKS_VULKAN_PARITY_DRIVER}"
            --left-traces "${PROJECT_BINARY_DIR}/open-loop-cpu-traces"
            --right-traces "${PROJECT_BINARY_DIR}/open-loop-vulkan-traces")
    set_tests_properties(motionbricks-open-loop-backend-parity PROPERTIES
        DEPENDS "motionbricks-open-loop-parity-cpu;motionbricks-open-loop-parity-vulkan"
        TIMEOUT 60)
endif()

if(MOTIONBRICKS_ENABLE_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR NOT MOTIONBRICKS_SANITIZE)
        message(FATAL_ERROR "Sampling/API fuzzing requires Clang and MOTIONBRICKS_SANITIZE=ON")
    endif()
    add_executable(motionbricks-sampling-fuzz sampling_fuzz.cpp)
    target_include_directories(motionbricks-sampling-fuzz PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(motionbricks-sampling-fuzz PRIVATE motionbricks_static)
    target_compile_options(motionbricks-sampling-fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(motionbricks-sampling-fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
    # Instrument code under test for coverage, not only the harness.
    target_compile_options(motionbricks_static PRIVATE -fsanitize=fuzzer-no-link)
    add_executable(motionbricks-sonic-fuzz sonic_fuzz.cpp)
    target_link_libraries(motionbricks-sonic-fuzz PRIVATE motionbricks_static)
    target_compile_options(motionbricks-sonic-fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(motionbricks-sonic-fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
    add_executable(motionbricks-inference-fuzz inference_fuzz.cpp)
    target_link_libraries(motionbricks-inference-fuzz PRIVATE motionbricks_static)
    target_compile_options(motionbricks-inference-fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(motionbricks-inference-fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

add_executable(motionbricks-sonic-api-test sonic_api_test.cpp)
target_link_libraries(motionbricks-sonic-api-test PRIVATE motionbricks_shared)
if(MOTIONBRICKS_ENABLE_PHYSICS)
    target_sources(motionbricks-sonic-api-test PRIVATE physics_collision_test.cpp)
    target_include_directories(motionbricks-sonic-api-test PRIVATE "${MUJOCO_INCLUDE_DIR}")
    target_link_libraries(motionbricks-sonic-api-test PRIVATE "${MUJOCO_LIBRARY}")
    target_compile_definitions(motionbricks-sonic-api-test PRIVATE MOTIONBRICKS_TEST_COLLISIONS=1)
endif()
add_test(NAME motionbricks-sonic-api COMMAND motionbricks-sonic-api-test)
set(MOTIONBRICKS_SONIC_GGUF "${PROJECT_SOURCE_DIR}/generated/sonic/ggml/sonic-g1.gguf" CACHE FILEPATH "Local SONIC GGUF; tests never download it")
set(MOTIONBRICKS_SONIC_FIXTURE "${PROJECT_SOURCE_DIR}/generated/sonic/ggml/all-layers.mbsonic" CACHE FILEPATH "Local independent upstream layer fixture")
set(MOTIONBRICKS_PHYSICS_SCENE "${PROJECT_SOURCE_DIR}/generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml" CACHE FILEPATH "Local SONIC physical scene")
set(MOTIONBRICKS_PHYSICS_CONFIG "${PROJECT_SOURCE_DIR}/generated/sonic/ggml/g1.mbphysics" CACHE FILEPATH "Local verified motor/geometry config")
if(MOTIONBRICKS_ENABLE_GGML AND TARGET ggml AND MOTIONBRICKS_ENABLE_PHYSICS AND EXISTS "${MOTIONBRICKS_SONIC_GGUF}" AND EXISTS "${MOTIONBRICKS_PHYSICS_SCENE}" AND EXISTS "${MOTIONBRICKS_PHYSICS_CONFIG}")
    add_test(NAME motionbricks-sonic-physics-api COMMAND motionbricks-sonic-api-test "${MOTIONBRICKS_SONIC_GGUF}" "${MOTIONBRICKS_PHYSICS_SCENE}" "${MOTIONBRICKS_PHYSICS_CONFIG}")
endif()
if(MOTIONBRICKS_ENABLE_GGML AND TARGET ggml AND EXISTS "${MOTIONBRICKS_SONIC_GGUF}" AND EXISTS "${MOTIONBRICKS_SONIC_FIXTURE}")
    add_test(NAME motionbricks-sonic-api-model COMMAND motionbricks-sonic-api-test "${MOTIONBRICKS_SONIC_GGUF}")
    add_test(NAME motionbricks-sonic-parity-cpu COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_GGUF}" "${MOTIONBRICKS_SONIC_FIXTURE}" cpu)
    if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        add_test(NAME motionbricks-sonic-parity-vulkan COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_GGUF}" "${MOTIONBRICKS_SONIC_FIXTURE}" vulkan)
        set_tests_properties(motionbricks-sonic-parity-vulkan PROPERTIES ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
    endif()
endif()

if(MOTIONBRICKS_SANITIZE)
    foreach(target
            motionbricks-inference-example
            motionbricks-inference-api-test
            motionbricks-inference-model-test
            motionbricks-sonic-api-test
            motionbricks-c-api-test
            motionbricks-cpp-wrapper-test
            motionbricks-session-replay-test
            motionbricks-plan-parity-format-test
            motionbricks-context-endpoints-test
            motionbricks-sampling-test
            motionbricks-agent-blend-test
            motionbricks-pose-parity-test
            motionbricks-decoder-parity-test
            motionbricks-root-parity-test
            motionbricks-agent-integration-test
            motionbricks-agent-backend-parity-test
            motionbricks-cli
            motionbricks-parity)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endforeach()
endif()
