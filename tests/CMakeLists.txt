if(MOTIONBRICKS_ENABLE_SOMA_CALIBRATED_TESTS AND NOT MOTIONBRICKS_ENABLE_GGML)
    message(FATAL_ERROR "SOMA calibrated parity tests require MOTIONBRICKS_ENABLE_GGML")
endif()

add_executable(motionbricks-c-api-test c_api_test.c)
target_link_libraries(motionbricks-c-api-test PRIVATE motionbricks::motionbricks)
target_compile_definitions(motionbricks-c-api-test PRIVATE MOTIONBRICKS_TEST_HAVE_GGML=$<BOOL:${MOTIONBRICKS_ENABLE_GGML}>)
add_test(NAME motionbricks-c-api COMMAND motionbricks-c-api-test)

if(MOTIONBRICKS_ENABLE_GGML)
    add_executable(motionbricks-soma-aligned-dot-test soma_aligned_dot_test.cpp)
    target_include_directories(motionbricks-soma-aligned-dot-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_features(motionbricks-soma-aligned-dot-test PRIVATE cxx_std_20)
    add_test(NAME motionbricks-soma-aligned-dot COMMAND motionbricks-soma-aligned-dot-test)
    add_executable(motionbricks-soma-welford-test soma_welford_test.cpp)
    target_include_directories(motionbricks-soma-welford-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_features(motionbricks-soma-welford-test PRIVATE cxx_std_20)
    add_test(NAME motionbricks-soma-welford COMMAND motionbricks-soma-welford-test)
    add_executable(motionbricks-soma-rsqrt-table-test soma_rsqrt_table_test.cpp)
    target_include_directories(motionbricks-soma-rsqrt-table-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_features(motionbricks-soma-rsqrt-table-test PRIVATE cxx_std_20)
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/rsqrt-mantissa-table/rsqrt-sm120-corrections.bin" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/rsqrt-scale-audit/pairs.bin")
        add_test(NAME motionbricks-soma-rsqrt-table COMMAND motionbricks-soma-rsqrt-table-test
            "${PROJECT_SOURCE_DIR}/generated/soma/rsqrt-mantissa-table/rsqrt-sm120-corrections.bin"
            "${PROJECT_SOURCE_DIR}/generated/soma/rsqrt-scale-audit/pairs.bin")
    endif()
    add_executable(motionbricks-soma-backend-softmax-probe soma_backend_softmax_probe.cpp)
    target_include_directories(motionbricks-soma-backend-softmax-probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-backend-softmax-probe PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-backend-softmax-probe PRIVATE motionbricks_static ggml)
    add_executable(motionbricks-soma-softmax-test soma_softmax_test.cpp)
    target_include_directories(motionbricks-soma-softmax-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_features(motionbricks-soma-softmax-test PRIVATE cxx_std_20)
    target_link_libraries(motionbricks-soma-softmax-test PRIVATE ggml)
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/exp2-fraction-table/exp2-sm120-corrections.bin" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-softmax-audit/softmax-reference.bin")
        add_test(NAME motionbricks-soma-softmax COMMAND motionbricks-soma-softmax-test
            "${PROJECT_SOURCE_DIR}/generated/soma/exp2-fraction-table/exp2-sm120-corrections.bin"
            "${PROJECT_SOURCE_DIR}/generated/soma/zombie-softmax-audit/softmax-reference.bin")
    endif()
    add_executable(motionbricks-soma-mixed-pose-probe soma_mixed_pose_probe.cpp)
    add_executable(motionbricks-soma-root-capacity-test soma_root_capacity_test.cpp)
    target_include_directories(motionbricks-soma-root-capacity-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-root-capacity-test PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-root-capacity-test PRIVATE motionbricks_static ggml)
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-fixtures/pose.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/root.gguf")
        foreach(precision fp32 mixed-fp16)
            add_test(NAME motionbricks-soma-root-capacity-${precision} COMMAND motionbricks-soma-root-capacity-test
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-fixtures/pose.gguf" 64 ${precision})
            set_tests_properties(motionbricks-soma-root-capacity-${precision} PROPERTIES RUN_SERIAL TRUE LABELS "capacity")
        endforeach()
    endif()
    add_executable(motionbricks-soma-mixed-feature-batch-probe soma_mixed_feature_batch_probe.cpp)
    target_include_directories(motionbricks-soma-mixed-feature-batch-probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-mixed-feature-batch-probe PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-mixed-feature-batch-probe PRIVATE motionbricks_static ggml)
    add_executable(motionbricks-soma-norm-probe soma_norm_probe.cpp)
    add_executable(motionbricks-soma-matmul-probe soma_matmul_probe.cpp)
    target_include_directories(motionbricks-soma-matmul-probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-matmul-probe PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-matmul-probe PRIVATE motionbricks_static ggml)
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            set(soma_projection_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-residue-audit/${motion}.gguf")
            if(EXISTS "${soma_projection_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                add_test(NAME motionbricks-soma-vulkan-prefix-${motion} COMMAND motionbricks-soma-matmul-probe
                    "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_projection_fixture}"
                    "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-prefix-${motion}.json" vulkan backend-prefix-matrix)
                set_tests_properties(motionbricks-soma-vulkan-prefix-${motion} PROPERTIES
                    RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                    ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
            endif()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(mlp_end start end)
                foreach(mlp_layer fc_layers.0 fc_layers.1 forward_projection)
                    set(soma_mlp_case "${motion}-${mlp_end}-${mlp_layer}")
                    set(soma_mlp_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-mlp-matrix-audit/${soma_mlp_case}.gguf")
                    if(EXISTS "${soma_mlp_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                        add_test(NAME motionbricks-soma-vulkan-mlp-${soma_mlp_case} COMMAND motionbricks-soma-matmul-probe
                            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_mlp_fixture}"
                            "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-mlp-${soma_mlp_case}.json" vulkan backend-prefix-matrix)
                        set_tests_properties(motionbricks-soma-vulkan-mlp-${soma_mlp_case} PROPERTIES
                            RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                            ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(layer qkv linear1)
                set(soma_nvjet3_case "${motion}-${layer}")
                set(soma_nvjet3_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-nvjet3-audit/${soma_nvjet3_case}.gguf")
                if(EXISTS "${soma_nvjet3_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-nvjet3-${soma_nvjet3_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_nvjet3_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-nvjet3-${soma_nvjet3_case}.json" vulkan backend-nvjet3)
                    set_tests_properties(motionbricks-soma-vulkan-nvjet3-${soma_nvjet3_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(layer linear2 cond2)
                set(soma_nvjet_split4_case "${motion}-${layer}")
                set(soma_nvjet_split4_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-nvjet-split4-audit/${soma_nvjet_split4_case}.gguf")
                if(EXISTS "${soma_nvjet_split4_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-nvjet-split4-${soma_nvjet_split4_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_nvjet_split4_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-nvjet-split4-${soma_nvjet_split4_case}.json" vulkan backend-nvjet-split4)
                    set_tests_properties(motionbricks-soma-vulkan-nvjet-split4-${soma_nvjet_split4_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(layer RANGE 0 15)
                set(soma_pose_scores_case "${motion}-layer${layer}")
                set(soma_pose_scores_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-pose-scores-audit/${soma_pose_scores_case}.gguf")
                if(EXISTS "${soma_pose_scores_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-pose-scores-${soma_pose_scores_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_pose_scores_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-pose-scores-${soma_pose_scores_case}.json" vulkan backend-nvjet-scores)
                    set_tests_properties(motionbricks-soma-vulkan-pose-scores-${soma_pose_scores_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(layer root-first root-split root-spatial root-pointwise)
                set(soma_root_conv_case "${motion}-${layer}")
                set(soma_root_conv_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-conv-audit/${soma_root_conv_case}.gguf")
                if(EXISTS "${soma_root_conv_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-conv-${soma_root_conv_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_root_conv_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-conv-${soma_root_conv_case}.json" vulkan backend-conv)
                    set_tests_properties(motionbricks-soma-vulkan-conv-${soma_root_conv_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(layer decoder-first decoder-split decoder-spatial decoder-pointwise)
                set(soma_decoder_conv_case "${motion}-${layer}")
                set(soma_decoder_conv_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-decoder-conv-audit/${soma_decoder_conv_case}.gguf")
                if(EXISTS "${soma_decoder_conv_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-decoder-conv-${soma_decoder_conv_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_decoder_conv_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-decoder-conv-${soma_decoder_conv_case}.json" vulkan backend-conv)
                    set_tests_properties(motionbricks-soma-vulkan-decoder-conv-${soma_decoder_conv_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            set(soma_nvjet8_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-nvjet8-audit/${motion}-outproj.gguf")
            if(EXISTS "${soma_nvjet8_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                add_test(NAME motionbricks-soma-vulkan-nvjet8-${motion} COMMAND motionbricks-soma-matmul-probe
                    "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${soma_nvjet8_fixture}"
                    "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-nvjet8-${motion}.json" vulkan backend-nvjet8-bias)
                set_tests_properties(motionbricks-soma-vulkan-nvjet8-${motion} PROPERTIES
                    RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                    ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
            endif()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(layer root-cond0 decoder-cond0 decoder-cond2)
                set(conditioning_cases "${motion}-${layer}")
                if(motion STREQUAL "walk" AND layer STREQUAL "root-cond0")
                    list(APPEND conditioning_cases "walk-root-cond0-eight")
                endif()
                if(layer STREQUAL "root-cond0")
                    set(conditioning_mode backend-split8-matrix)
                else()
                    set(conditioning_mode backend-prefix-matrix)
                endif()
                foreach(conditioning_case IN LISTS conditioning_cases)
                    set(conditioning_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-conditioning-audit/${conditioning_case}.gguf")
                    if(EXISTS "${conditioning_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                        add_test(NAME motionbricks-soma-vulkan-conditioning-${conditioning_case} COMMAND motionbricks-soma-matmul-probe
                            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${conditioning_fixture}"
                            "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-conditioning-${conditioning_case}.json" vulkan ${conditioning_mode})
                        set_tests_properties(motionbricks-soma-vulkan-conditioning-${conditioning_case} PROPERTIES
                            RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                            ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            set(target_cases pose-input pose-eight target0 target2)
            if(motion STREQUAL "zombie")
                list(APPEND target_cases step1-target2)
            endif()
            foreach(target_case IN LISTS target_cases)
                set(target_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-target-audit/${motion}-${target_case}.gguf")
                if(EXISTS "${target_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-target-${motion}-${target_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${target_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-target-${motion}-${target_case}.json" vulkan backend-prefix-matrix)
                    set_tests_properties(motionbricks-soma-vulkan-target-${motion}-${target_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        foreach(motion walk sit zombie)
            foreach(pose_matrix_case qkv outproj linear1 linear2)
                set(pose_matrix_fixture "${PROJECT_SOURCE_DIR}/generated/soma/vulkan-pose-matrix-audit/${motion}-${pose_matrix_case}-eight.gguf")
                if(pose_matrix_case STREQUAL "outproj")
                    set(pose_matrix_mode backend-prefix-bias)
                else()
                    set(pose_matrix_mode backend-prefix-matrix)
                endif()
                if(EXISTS "${pose_matrix_fixture}" AND EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
                    add_test(NAME motionbricks-soma-vulkan-pose-matrix-${motion}-${pose_matrix_case} COMMAND motionbricks-soma-matmul-probe
                        "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${pose_matrix_fixture}"
                        "${CMAKE_CURRENT_BINARY_DIR}/soma-vulkan-pose-matrix-${motion}-${pose_matrix_case}.json" vulkan ${pose_matrix_mode})
                    set_tests_properties(motionbricks-soma-vulkan-pose-matrix-${motion}-${pose_matrix_case} PROPERTIES
                        RUN_SERIAL TRUE PASS_REGULAR_EXPRESSION "max=0 mismatches=0"
                        ENVIRONMENT "MOTIONBRICKS_SOMA_NATIVE_F16=1;GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
                endif()
            endforeach()
        endforeach()
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-token-qkv.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
        add_test(NAME motionbricks-soma-nvjet-qkv COMMAND motionbricks-soma-matmul-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-token-qkv.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-nvjet-qkv.json" cpu nvjet-qkv-graph)
        set_tests_properties(motionbricks-soma-nvjet-qkv PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-token-outproj.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
        add_test(NAME motionbricks-soma-nvjet-outproj COMMAND motionbricks-soma-matmul-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-token-outproj.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-nvjet-outproj.json" cpu nvjet-outproj-graph)
        set_tests_properties(motionbricks-soma-nvjet-outproj PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-token-linear2.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
        add_test(NAME motionbricks-soma-nvjet-split4 COMMAND motionbricks-soma-matmul-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-token-linear2.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-nvjet-split4.json" cpu nvjet-split4-graph)
        set_tests_properties(motionbricks-soma-nvjet-split4 PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-external-cond2.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
        add_test(NAME motionbricks-soma-nvjet-cond2 COMMAND motionbricks-soma-matmul-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/zombie-root-external-cond2.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-nvjet-cond2.json" cpu nvjet-split4-graph)
        set_tests_properties(motionbricks-soma-nvjet-cond2 PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-pose-score.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
        add_test(NAME motionbricks-soma-nvjet-scores COMMAND motionbricks-soma-matmul-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/zombie-pose-score.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-nvjet-scores.json" cpu nvjet-scores-graph)
        set_tests_properties(motionbricks-soma-nvjet-scores PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/sit-root-attention-values.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
        add_test(NAME motionbricks-soma-attention-tail COMMAND motionbricks-soma-matmul-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/sit-root-attention-values.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-attention-tail.json" cpu aligned16-tail-graph)
        set_tests_properties(motionbricks-soma-attention-tail PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
    endif()
    foreach(conv_case first split spatial pointwise)
        if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/zombie-decoder-conv-${conv_case}.gguf" AND
           EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
            add_test(NAME motionbricks-soma-conv-${conv_case} COMMAND motionbricks-soma-matmul-probe
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
                "${PROJECT_SOURCE_DIR}/generated/soma/zombie-decoder-conv-${conv_case}.gguf"
                "${CMAKE_CURRENT_BINARY_DIR}/soma-conv-${conv_case}.json" cpu conv-profiled-graph)
            set_tests_properties(motionbricks-soma-conv-${conv_case} PROPERTIES PASS_REGULAR_EXPRESSION "max=0 mismatches=0")
        endif()
    endforeach()
    target_include_directories(motionbricks-soma-norm-probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-norm-probe PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-norm-probe PRIVATE motionbricks_static ggml)
    target_include_directories(motionbricks-soma-mixed-pose-probe PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-mixed-pose-probe PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-mixed-pose-probe PRIVATE motionbricks_static ggml)
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-mixed-pose.gguf" AND
       EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/pose.gguf")
        # Tests arithmetic boundaries only; the probe also reports numerical
        # differences, which are not a full mixed-FP16 parity acceptance gate.
        add_test(NAME motionbricks-soma-precision-contract COMMAND motionbricks-soma-mixed-pose-probe
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-mixed-pose.gguf"
            "${CMAKE_CURRENT_BINARY_DIR}/soma-precision-contract.json" cpu)
    endif()
    add_executable(motionbricks-soma-api-parity-test soma_api_parity_test.cpp)
    target_link_libraries(motionbricks-soma-api-parity-test PRIVATE motionbricks_shared ggml)
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-io/playback.gguf")
        add_test(NAME motionbricks-soma-api-parity COMMAND motionbricks-soma-api-parity-test
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-io/playback.gguf")
        if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
            add_test(NAME motionbricks-soma-api-parity-vulkan COMMAND motionbricks-soma-api-parity-test
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-io/playback.gguf" vulkan)
            set_tests_properties(motionbricks-soma-api-parity-vulkan PROPERTIES RUN_SERIAL TRUE)
        endif()
    endif()
    add_executable(motionbricks-soma-batch-parity-test soma_batch_parity_test.cpp)
    add_executable(motionbricks-soma-feature-batch-parity-test soma_feature_batch_parity_test.cpp)
    add_executable(motionbricks-soma-motion-batch-parity-test soma_motion_batch_parity_test.cpp)
    target_include_directories(motionbricks-soma-motion-batch-parity-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-motion-batch-parity-test PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-motion-batch-parity-test PRIVATE motionbricks_static ggml)
    if(MOTIONBRICKS_ENABLE_SOMA_CALIBRATED_TESTS)
        set(soma_calibration_root "${PROJECT_SOURCE_DIR}/generated/soma")
        set(soma_rsqrt "${soma_calibration_root}/rsqrt-mantissa-table/rsqrt-sm120-corrections.bin")
        set(soma_exp2 "${soma_calibration_root}/exp2-fraction-table/exp2-sm120-corrections.bin")
        set(soma_motion_fixtures)
        foreach(chain chain-walk-half-cpu chain-sit-half-cpu-coverage chain-zombie-half-cpu-coverage)
            list(APPEND soma_motion_fixtures "${soma_calibration_root}/${chain}/step-000/playback/playback.gguf")
        endforeach()
        foreach(asset "${soma_rsqrt}" "${soma_exp2}" ${soma_motion_fixtures}
                      "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/manifest.json")
            if(NOT EXISTS "${asset}")
                message(FATAL_ERROR "SOMA calibrated parity requires ${asset}; see reference/README.md")
            endif()
        endforeach()
        foreach(chain walk sit zombie)
            if(chain STREQUAL "walk")
                set(soma_api_chain "chain-walk-half-cpu")
            else()
                set(soma_api_chain "chain-${chain}-half-cpu-coverage")
            endif()
            add_test(NAME motionbricks-soma-api-calibrated-${chain} COMMAND motionbricks-soma-api-parity-test
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
                "${soma_calibration_root}/${soma_api_chain}/step-000/playback/playback.gguf"
                cpu "${soma_rsqrt}" "${soma_exp2}")
            set_tests_properties(motionbricks-soma-api-calibrated-${chain} PROPERTIES RUN_SERIAL TRUE)
        endforeach()
        add_test(NAME motionbricks-soma-motion-batch-calibrated COMMAND motionbricks-soma-motion-batch-parity-test
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" cpu mixed-fp16 ${soma_motion_fixtures})
        set_tests_properties(motionbricks-soma-motion-batch-calibrated PROPERTIES RUN_SERIAL TRUE
            ENVIRONMENT "MOTIONBRICKS_SOMA_ALIGNED_MATMUL_TRIAL=1;MOTIONBRICKS_SOMA_WELFORD_TRIAL=1;MOTIONBRICKS_SOMA_RSQRT_TABLE=${soma_rsqrt};MOTIONBRICKS_SOMA_EXP2_TABLE=${soma_exp2}")
    endif()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-io/playback.gguf" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-io/playback.gguf" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/height-stress-low-io/playback.gguf")
        add_test(NAME motionbricks-soma-motion-batch-parity COMMAND motionbricks-soma-motion-batch-parity-test
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" cpu
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-io/playback.gguf"
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-io/playback.gguf"
            "${PROJECT_SOURCE_DIR}/generated/soma/height-stress-low-io/playback.gguf")
        if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
            add_test(NAME motionbricks-soma-motion-batch-parity-vulkan COMMAND motionbricks-soma-motion-batch-parity-test
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" vulkan
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-io/playback.gguf"
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-io/playback.gguf"
                "${PROJECT_SOURCE_DIR}/generated/soma/height-stress-low-io/playback.gguf")
            set_tests_properties(motionbricks-soma-motion-batch-parity-vulkan PROPERTIES RUN_SERIAL TRUE)
        endif()
    endif()
    target_include_directories(motionbricks-soma-feature-batch-parity-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-feature-batch-parity-test PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-feature-batch-parity-test PRIVATE motionbricks_static ggml)
    target_include_directories(motionbricks-soma-batch-parity-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-batch-parity-test PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-batch-parity-test PRIVATE motionbricks_static ggml)
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-fixtures/pose.gguf" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-four16-fixtures/pose.gguf")
        add_test(NAME motionbricks-soma-batch-parity COMMAND motionbricks-soma-batch-parity-test
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}"
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-fixtures/pose.gguf"
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-four16-fixtures/pose.gguf")
        if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-fixtures/pose.gguf")
            add_test(NAME motionbricks-soma-feature-batch-parity COMMAND motionbricks-soma-feature-batch-parity-test
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" cpu
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-fixtures/pose.gguf"
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-four16-fixtures/pose.gguf"
                "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-fixtures/pose.gguf")
            if(MOTIONBRICKS_ENABLE_VULKAN AND MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
                add_test(NAME motionbricks-soma-feature-batch-parity-vulkan COMMAND motionbricks-soma-feature-batch-parity-test
                    "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" vulkan
                    "${PROJECT_SOURCE_DIR}/generated/soma/walk-one-fixtures/pose.gguf"
                    "${PROJECT_SOURCE_DIR}/generated/soma/walk-four16-fixtures/pose.gguf"
                    "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-fixtures/pose.gguf")
                set_tests_properties(motionbricks-soma-feature-batch-parity-vulkan PROPERTIES RUN_SERIAL TRUE)
            endif()
        endif()
    endif()
    add_executable(motionbricks-soma-run "${PROJECT_SOURCE_DIR}/src/soma_runner.cpp")
    target_include_directories(motionbricks-soma-run PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(motionbricks-soma-run PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-run PRIVATE motionbricks_static ggml)
    add_executable(motionbricks-soma-pose-parity-test soma_pose_parity_test.cpp)
    add_executable(motionbricks-soma-playback-parity-test soma_playback_parity_test.cpp)
    target_include_directories(motionbricks-soma-playback-parity-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(motionbricks-soma-playback-parity-test PRIVATE motionbricks_static ggml)
    foreach(chain IN ITEMS chain-walk-half-cpu chain-sit-half-cpu-coverage chain-zombie-half-cpu-coverage)
        foreach(step IN ITEMS 000 001 002)
            set(encoder_fixture "${PROJECT_SOURCE_DIR}/generated/soma/${chain}/step-${step}/playback/playback.gguf")
            if(EXISTS "${encoder_fixture}")
                add_test(NAME motionbricks-soma-encoder-${chain}-${step}
                    COMMAND motionbricks-soma-playback-parity-test "${encoder_fixture}" --exact-encoding)
                add_test(NAME motionbricks-soma-playback-${chain}-${step}
                    COMMAND motionbricks-soma-playback-parity-test "${encoder_fixture}" --exact-playback)
            endif()
        endforeach()
    endforeach()
    if(EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-io/playback.gguf")
        add_test(NAME motionbricks-soma-playback-parity COMMAND motionbricks-soma-playback-parity-test
            "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-io/playback.gguf")
    endif()
    if(EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/pose.gguf" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-fixtures/pose.gguf")
        add_test(NAME motionbricks-soma-walk-parity COMMAND motionbricks-soma-pose-parity-test
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${PROJECT_SOURCE_DIR}/generated/soma/walk-soma-fixtures/pose.gguf")
    endif()
    target_include_directories(motionbricks-soma-pose-parity-test PRIVATE
        "${PROJECT_SOURCE_DIR}/src" "${MOTIONBRICKS_GGML_SOURCE_DIR}/include")
    target_compile_definitions(motionbricks-soma-pose-parity-test PRIVATE MOTIONBRICKS_HAVE_GGML=1)
    target_link_libraries(motionbricks-soma-pose-parity-test PRIVATE motionbricks_static ggml)
    foreach(height_case low high)
        set(height_fixture "${PROJECT_SOURCE_DIR}/generated/soma/height-stress-${height_case}-fixtures/pose.gguf")
        if(EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/pose.gguf" AND EXISTS "${height_fixture}")
            add_test(NAME motionbricks-soma-height-${height_case}-parity COMMAND motionbricks-soma-pose-parity-test
                "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${height_fixture}")
        endif()
    endforeach()
    if(EXISTS "${MOTIONBRICKS_SOMA_TEST_BUNDLE}/pose.gguf" AND
       EXISTS "${PROJECT_SOURCE_DIR}/generated/soma/fixtures/pose.gguf")
        add_test(NAME motionbricks-soma-pose-parity COMMAND motionbricks-soma-pose-parity-test
            "${MOTIONBRICKS_SOMA_TEST_BUNDLE}" "${PROJECT_SOURCE_DIR}/generated/soma/fixtures/pose.gguf")
    endif()
endif()

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
    add_executable(motionbricks-batch-model-test batch_model_test.cpp)
    target_include_directories(motionbricks-batch-model-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(motionbricks-batch-model-test PRIVATE motionbricks_static)
    add_test(NAME motionbricks-batch-model COMMAND motionbricks-batch-model-test
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
    set(MOTIONBRICKS_SOMA_REFERENCE_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH
        "Python interpreter with numpy and torch for SOMA reference-tool tests")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env --unset=PYTHONPATH
            "${MOTIONBRICKS_SOMA_REFERENCE_PYTHON}" -c "import numpy, torch"
        RESULT_VARIABLE MOTIONBRICKS_SOMA_TEST_DEPS
        OUTPUT_QUIET ERROR_QUIET)
    if(MOTIONBRICKS_SOMA_TEST_DEPS EQUAL 0)
        add_test(NAME motionbricks-reference-soma-tools
            COMMAND "${CMAKE_COMMAND}" -E env --unset=PYTHONPATH
                "${MOTIONBRICKS_SOMA_REFERENCE_PYTHON}" -m unittest test_soma_tools.py)
        set_tests_properties(motionbricks-reference-soma-tools PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/reference")
    endif()
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
# These checks invoke the ABI inside assert(); keep them active in CPU release builds.
if(MSVC)
    target_compile_options(motionbricks-sonic-api-test PRIVATE /UNDEBUG)
else()
    target_compile_options(motionbricks-sonic-api-test PRIVATE -UNDEBUG)
endif()
if(MOTIONBRICKS_ENABLE_PHYSICS)
    target_sources(motionbricks-sonic-api-test PRIVATE physics_collision_test.cpp)
    target_include_directories(motionbricks-sonic-api-test PRIVATE "${MUJOCO_INCLUDE_DIR}")
    target_link_libraries(motionbricks-sonic-api-test PRIVATE "${MUJOCO_LIBRARY}")
    target_compile_definitions(motionbricks-sonic-api-test PRIVATE MOTIONBRICKS_TEST_COLLISIONS=1)
    if(UNIX AND NOT APPLE)
        add_executable(motionbricks-mujoco-abi-test mujoco_abi_test.cpp)
        target_link_libraries(motionbricks-mujoco-abi-test PRIVATE motionbricks_shared)
        target_link_options(motionbricks-mujoco-abi-test PRIVATE "-Wl,--export-dynamic-symbol=mj_version")
        if(MOTIONBRICKS_SANITIZE)
            target_compile_options(motionbricks-mujoco-abi-test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(motionbricks-mujoco-abi-test PRIVATE -fsanitize=address,undefined)
        endif()
        add_test(NAME motionbricks-mujoco-abi-mismatch COMMAND motionbricks-mujoco-abi-test)
    endif()
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
    add_test(NAME motionbricks-sonic-parity-cpu COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_GGUF}" "${MOTIONBRICKS_SONIC_FIXTURE}" cpu 1)
    add_test(NAME motionbricks-sonic-parity-cpu-2 COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_GGUF}" "${MOTIONBRICKS_SONIC_FIXTURE}" cpu 2)
    if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
        add_test(NAME motionbricks-sonic-parity-vulkan COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_GGUF}" "${MOTIONBRICKS_SONIC_FIXTURE}" vulkan)
        set_tests_properties(motionbricks-sonic-parity-vulkan PROPERTIES ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
    endif()
endif()

set(MOTIONBRICKS_SONIC_SMPL_GGUF "${PROJECT_SOURCE_DIR}/generated/sonic/ggml/sonic-g1-smpl.gguf" CACHE FILEPATH "Optional combined G1/SMPL SONIC GGUF")
set(MOTIONBRICKS_SONIC_SMPL_FIXTURE "${PROJECT_SOURCE_DIR}/generated/sonic/ggml/smpl-layers.mbsonic" CACHE FILEPATH "Independent SMPL ONNX layer fixture")
add_executable(motionbricks-sonic-smpl-test sonic_smpl_test.cpp)
target_link_libraries(motionbricks-sonic-smpl-test PRIVATE motionbricks_shared)
if(MSVC)
    target_compile_options(motionbricks-sonic-smpl-test PRIVATE /UNDEBUG)
else()
    target_compile_options(motionbricks-sonic-smpl-test PRIVATE -UNDEBUG)
endif()
if(MOTIONBRICKS_ENABLE_GGML AND TARGET ggml AND EXISTS "${MOTIONBRICKS_SONIC_SMPL_GGUF}")
    if(EXISTS "${MOTIONBRICKS_SONIC_GGUF}")
        add_test(NAME motionbricks-sonic-smpl-api COMMAND motionbricks-sonic-smpl-test "${MOTIONBRICKS_SONIC_SMPL_GGUF}" "${MOTIONBRICKS_SONIC_GGUF}")
        if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
            add_test(NAME motionbricks-sonic-smpl-api-vulkan COMMAND motionbricks-sonic-smpl-test "${MOTIONBRICKS_SONIC_SMPL_GGUF}" "${MOTIONBRICKS_SONIC_GGUF}" vulkan)
            set_tests_properties(motionbricks-sonic-smpl-api-vulkan PROPERTIES ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
        endif()
    endif()
    if(EXISTS "${MOTIONBRICKS_SONIC_SMPL_FIXTURE}")
        add_test(NAME motionbricks-sonic-smpl-parity-cpu COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_SMPL_GGUF}" "${MOTIONBRICKS_SONIC_SMPL_FIXTURE}" cpu 1)
        add_test(NAME motionbricks-sonic-smpl-parity-cpu-2 COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_SMPL_GGUF}" "${MOTIONBRICKS_SONIC_SMPL_FIXTURE}" cpu 2)
        if(MOTIONBRICKS_ENABLE_VULKAN_PARITY_TESTS)
            add_test(NAME motionbricks-sonic-smpl-parity-vulkan COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_SMPL_GGUF}" "${MOTIONBRICKS_SONIC_SMPL_FIXTURE}" vulkan)
            set_tests_properties(motionbricks-sonic-smpl-parity-vulkan PROPERTIES ENVIRONMENT "GGML_VK_VISIBLE_DEVICES=${MOTIONBRICKS_VULKAN_PARITY_DEVICE}")
        endif()
    endif()
    if(EXISTS "${MOTIONBRICKS_SONIC_FIXTURE}")
        add_test(NAME motionbricks-sonic-combined-g1-parity COMMAND motionbricks-sonic-parity "${MOTIONBRICKS_SONIC_SMPL_GGUF}" "${MOTIONBRICKS_SONIC_FIXTURE}" cpu 1)
    endif()
endif()

if(MOTIONBRICKS_CPU_ALL_VARIANTS AND TARGET ggml AND NOT MOTIONBRICKS_SANITIZE AND
   CMAKE_SYSTEM_NAME STREQUAL "Linux" AND Python3_Interpreter_FOUND AND
   EXISTS "${MOTIONBRICKS_SONIC_GGUF}")
    add_test(NAME motionbricks-cpu-backend-loading
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/backend_loading_test.py"
            --library "$<TARGET_FILE:motionbricks_shared>"
            --model "${MOTIONBRICKS_SONIC_GGUF}")
    set_tests_properties(motionbricks-cpu-backend-loading PROPERTIES TIMEOUT 120)
endif()

if(MOTIONBRICKS_SANITIZE)
    foreach(target
            motionbricks-inference-example
            motionbricks-inference-api-test
            motionbricks-inference-model-test
            motionbricks-sonic-api-test
            motionbricks-sonic-smpl-test
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

add_executable(motionbricks-retarget-api-test retarget_api_test.cpp)
target_include_directories(motionbricks-retarget-api-test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(motionbricks-retarget-api-test PRIVATE motionbricks_static)
add_test(NAME motionbricks-retarget-api COMMAND motionbricks-retarget-api-test)
if(MOTIONBRICKS_ENABLE_FUZZERS)
    add_executable(motionbricks-retarget-fuzz retarget_fuzz.cpp)
    target_link_libraries(motionbricks-retarget-fuzz PRIVATE motionbricks_static)
    target_compile_options(motionbricks-retarget-fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(motionbricks-retarget-fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
if(MOTIONBRICKS_SANITIZE)
    target_compile_options(motionbricks-retarget-api-test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(motionbricks-retarget-api-test PRIVATE -fsanitize=address,undefined)
endif()
