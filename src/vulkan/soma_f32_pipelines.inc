    // Preserve FP32 operands while retaining cooperative FP16 matrix kernels.
    // Included inside ggml_vk_load_shaders with the scalar CREATE_MM macro.
    // Tile geometry matches the pinned upstream scalar FP32 path.
    if (device->coopmat2) {
        const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;
        l_warptile = {128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8};
        m_warptile = {128, 64, 64, 16, subgroup_size_8, 32, 2, 4, 2, 1, subgroup_size_8};
        s_warptile = {subgroup_size_32, 32, 32, 16, s_warptile_wm, 32, 2, 2, 2, 1, subgroup_size_8};
        l_wg_denoms = {128, 128, 1};
        m_wg_denoms = {64, 64, 1};
        s_wg_denoms = {32, 32, 1};
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
    }
