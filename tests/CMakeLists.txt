add_executable(motionbricks-c-api-test c_api_test.c)
target_link_libraries(motionbricks-c-api-test PRIVATE motionbricks::motionbricks)
add_test(NAME motionbricks-c-api COMMAND motionbricks-c-api-test)

add_executable(motionbricks-cpp-wrapper-test cpp_wrapper_test.cpp)
target_link_libraries(motionbricks-cpp-wrapper-test PRIVATE motionbricks::motionbricks)
add_test(NAME motionbricks-cpp-wrapper COMMAND motionbricks-cpp-wrapper-test)

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
        add_test(NAME motionbricks-go-demo
            COMMAND "${CMAKE_COMMAND}" -E env
                "CGO_ENABLED=0"
                "MOTIONBRICKS_LIB=$<TARGET_FILE:motionbricks_shared>"
                "MOTIONBRICKS_MODEL=${MOTIONBRICKS_REFERENCE_BUNDLE}"
                "MOTIONBRICKS_STYLES=${MOTIONBRICKS_REFERENCE_STYLES}"
                "MOTIONBRICKS_CHROME=${MOTIONBRICKS_CHROMIUM_EXECUTABLE}"
                "${MOTIONBRICKS_GO_EXECUTABLE}" test -v ./...)
        set_tests_properties(motionbricks-go-demo PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../demo"
            RUN_SERIAL TRUE
            TIMEOUT 60)
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

if(MOTIONBRICKS_SANITIZE)
    foreach(target
            motionbricks-c-api-test
            motionbricks-cpp-wrapper-test
            motionbricks-pose-parity-test
            motionbricks-decoder-parity-test
            motionbricks-root-parity-test
            motionbricks-agent-integration-test
            motionbricks-agent-backend-parity-test
            motionbricks-cli)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endforeach()
endif()
