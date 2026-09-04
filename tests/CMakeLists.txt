add_executable(motionbricks-c-api-test c_api_test.c)
target_link_libraries(motionbricks-c-api-test PRIVATE motionbricks::motionbricks)
add_test(NAME motionbricks-c-api COMMAND motionbricks-c-api-test)

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

if(MOTIONBRICKS_SANITIZE)
    foreach(target
            motionbricks-c-api-test
            motionbricks-cpp-wrapper-test
            motionbricks-session-replay-test
            motionbricks-plan-parity-format-test
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
