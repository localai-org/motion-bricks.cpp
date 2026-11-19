# The cooperative-matrix backend normally stages FP32 matrices as FP16. Keep
# scalar FP32 pipelines available so enabling SOMA mixed precision does not
# silently change ordinary FP32 inference in the same process.
function(motionbricks_patch_soma_vulkan_backend overlay)
    set(source "${overlay}/src/ggml-vulkan/ggml-vulkan.cpp")
    file(SHA256 "${source}" backend_hash)
    if(NOT backend_hash STREQUAL "c73e8f7980cd416f7fd30860c43f85e4ecacb7db84c0bf057cbb505aaf90142f")
        message(FATAL_ERROR "SOMA Vulkan backend overlay requires the validated pinned GGML backend source")
    endif()
    file(READ "${source}" backend)
    string(REPLACE "#undef CREATE_MM\n\n    // mul mat vec"
        "#include \"soma_f32_pipelines.inc\"\n#undef CREATE_MM\n\n    // mul mat vec" backend "${backend}")
    string(REPLACE
        "if (ctx->device->coopmat2) {\n        const uint32_t shader_core_count"
        "if (ctx->device->coopmat2 && src0_type != GGML_TYPE_F32) {\n        const uint32_t shader_core_count"
        backend "${backend}")
    set(original [=[    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);]=])
    set(replacement [=[    // Preserve contiguous FP32 x FP32 operands; explicit half operands retain
    // the upstream cooperative-matrix path. Non-contiguous staging is unchanged.
    const bool x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32 && src1->type != GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32 && src0->type != GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);]=])
    string(REPLACE "${original}" "${replacement}" backend "${backend}")
    string(REPLACE "static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {"
        "#include \"soma_capabilities.inc\"\n\nstatic const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {"
        backend "${backend}")
    string(REPLACE "/* .get_proc_address = */ NULL,"
        "/* .get_proc_address = */ motionbricks_soma_vulkan_get_proc_address,"
        backend "${backend}")
    file(WRITE "${source}" "${backend}")
    configure_file("${PROJECT_SOURCE_DIR}/src/vulkan/soma_f32_pipelines.inc"
        "${overlay}/src/ggml-vulkan/soma_f32_pipelines.inc" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/src/vulkan/soma_capabilities.inc"
        "${overlay}/src/ggml-vulkan/soma_capabilities.inc" COPYONLY)
endfunction()
