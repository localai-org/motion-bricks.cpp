#pragma once
#include "neural_precision.hpp"
#include <string>
#include <stdexcept>

namespace motionbricks::detail {
// Explicit diagnostic configuration shared by single and batch parity runners.
// Public API loading does not read these trial environment variables.
inline void configure_soma_trial(const std::string & backend,bool mixed_fp16) {
    const char * backend_softmax=std::getenv("MOTIONBRICKS_SOMA_VULKAN_SOFTMAX_TRIAL");
    neural_backend_softmax_trial=backend_softmax && std::strcmp(backend_softmax,"1")==0;
    if(neural_backend_softmax_trial && (backend!="vulkan" || !mixed_fp16))
        throw std::runtime_error("Scoped Vulkan softmax requires Vulkan mixed-fp16");
#if !defined(MOTIONBRICKS_HAVE_SOMA_VULKAN_SHADERS)
    if(neural_backend_softmax_trial)throw std::runtime_error("Scoped Vulkan softmax requires MOTIONBRICKS_ENABLE_SOMA_VULKAN_SHADERS");
#endif
    const char * nvjet3_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_NVJET3_TRIAL");
    neural_backend_nvjet3_trial=nvjet3_trial && std::strcmp(nvjet3_trial,"1")==0;
    if(neural_backend_nvjet3_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan NVJET3 trial requires Vulkan mixed-fp16 and native half operands");
    const char * nvjet8_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_NVJET8_TRIAL");
    neural_backend_nvjet8_trial=nvjet8_trial && std::strcmp(nvjet8_trial,"1")==0;
    if(neural_backend_nvjet8_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan NVJET8 trial requires Vulkan mixed-fp16 and native half operands");
    const char * pose_scores_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_POSE_SCORES_TRIAL");
    neural_backend_pose_scores_trial=pose_scores_trial && std::strcmp(pose_scores_trial,"1")==0;
    if(neural_backend_pose_scores_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan POSE_SCORES trial requires Vulkan mixed-fp16 and native half operands");
    const char * root_conv_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_ROOT_CONV_TRIAL");
    neural_backend_root_conv_trial=root_conv_trial && std::strcmp(root_conv_trial,"1")==0;
    if(neural_backend_root_conv_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan ROOT_CONV trial requires Vulkan mixed-fp16 and native half operands");
    const char * conditioning_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_CONDITIONING_TRIAL");
    neural_backend_conditioning_trial=conditioning_trial && std::strcmp(conditioning_trial,"1")==0;
    if(neural_backend_conditioning_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan CONDITIONING trial requires Vulkan mixed-fp16 and native half operands");
    const char * decoder_conv_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_DECODER_CONV_TRIAL");
    neural_backend_decoder_conv_trial=decoder_conv_trial && std::strcmp(decoder_conv_trial,"1")==0;
    if(neural_backend_decoder_conv_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan DECODER_CONV trial requires Vulkan mixed-fp16 and native half operands");
    const char * nvjet_split4_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_NVJET_SPLIT4_TRIAL");
    neural_backend_nvjet_split4_trial=nvjet_split4_trial && std::strcmp(nvjet_split4_trial,"1")==0;
    if(neural_backend_nvjet_split4_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan NVJET_SPLIT4 trial requires Vulkan mixed-fp16 and native half operands");
    const char * norm_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_NORM_TRIAL");
    neural_backend_norm_trial=norm_trial && std::strcmp(norm_trial,"1")==0;
    if(neural_backend_norm_trial && (backend!="vulkan" || !mixed_fp16))
        throw std::runtime_error("Vulkan norm trial requires Vulkan mixed-fp16 and experimental norm shader");
    const char * welford_trial=std::getenv("MOTIONBRICKS_SOMA_WELFORD_TRIAL");
    neural_welford_trial=welford_trial && std::strcmp(welford_trial,"1")==0;
    if(neural_welford_trial && (backend!="cpu" || !mixed_fp16))
        throw std::runtime_error("Welford trial requires CPU mixed-fp16");
    if(const char * table=std::getenv("MOTIONBRICKS_SOMA_RSQRT_TABLE")) {
        if(!neural_welford_trial)throw std::runtime_error("rsqrt calibration requires the CPU Welford trial");
        load_soma_rsqrt_table(table);
    }
    if(const char * table=std::getenv("MOTIONBRICKS_SOMA_EXP2_TABLE")) {
        if(backend!="cpu" || !mixed_fp16)throw std::runtime_error("exp2 calibration requires CPU mixed-fp16");
        load_soma_exp2_table(table);
        neural_softmax_trial=true;
    }
    const char * aligned_trial=std::getenv("MOTIONBRICKS_SOMA_ALIGNED_ROOT_TRIAL");
    neural_aligned_root_trial=aligned_trial && std::strcmp(aligned_trial,"1")==0;
    const char * aligned_matmul_trial=std::getenv("MOTIONBRICKS_SOMA_ALIGNED_MATMUL_TRIAL");
    neural_aligned_matmul_trial=aligned_matmul_trial && std::strcmp(aligned_matmul_trial,"1")==0;
    neural_aligned_root_trial=neural_aligned_root_trial || neural_aligned_matmul_trial;
    if(neural_aligned_root_trial && (backend!="cpu" || !mixed_fp16))
        throw std::runtime_error("aligned root trial requires CPU mixed-fp16");
    const char * prefix_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_PREFIX_TRIAL");
    neural_backend_prefix_trial=prefix_trial && std::strcmp(prefix_trial,"1")==0;
    if(neural_backend_prefix_trial && (backend!="vulkan" || !mixed_fp16 || !native_half_operands_requested()))
        throw std::runtime_error("Vulkan prefix trial requires Vulkan mixed-fp16 and MOTIONBRICKS_SOMA_NATIVE_F16=1");
    const char * backend_split_trial=std::getenv("MOTIONBRICKS_SOMA_VULKAN_SPLIT_ROOT_TRIAL");
    neural_backend_split_root_trial=backend_split_trial && std::strcmp(backend_split_trial,"1")==0;
    if(neural_backend_split_root_trial && (backend!="vulkan" || !mixed_fp16 ||
       !native_half_operands_requested() || neural_aligned_root_trial))
        throw std::runtime_error("Vulkan split trial requires Vulkan mixed-fp16 and MOTIONBRICKS_SOMA_NATIVE_F16=1");
}
}
