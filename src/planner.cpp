#include "planner.hpp"

#include "decoder.hpp"
#include "handles.hpp"
#include "neural_runtime.hpp"
#include "pose.hpp"
#include "root.hpp"
#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <limits>
#include <vector>

namespace motionbricks::detail {
namespace {

float wrapped(float value) {
    constexpr float pi = 3.14159265358979323846F;
    return std::remainder(value, 2.0F * pi);
}

} // namespace

mb_status run_transition(const mb_model & model,
                         const transition_constraints & constraints,
                         mb_motion & output,
                         std::uint32_t * selected_tokens,
                         transition_trace * trace,
                         std::string & reason, std::uint64_t seed, bool sampling_argmax) {
    if (!model.runtime || model.motion_mean.size() != 418U || model.motion_std.size() != 418U) {
        reason = "model is missing neural or normalization data";
        return MB_INCOMPATIBLE_MODEL;
    }
    if (trace != nullptr) {
        *trace = {};
        trace->constraints = constraints;
    }
    const float initial_x = constraints.global_root[0];
    const float initial_z = constraints.global_root[2];
    const float initial_heading = std::atan2(constraints.global_root[4], constraints.global_root[3]);

    std::vector<float> global(8U * global_root_width);
    std::vector<float> local(8U * local_root_width);
    std::vector<float> poses(8U * internal_pose_width);
    for (std::uint32_t frame = 0; frame < 8U; ++frame) {
        for (std::uint32_t feature = 0; feature < global_root_width; ++feature) {
            float value = constraints.global_root[frame * global_root_width + feature];
            if (feature == 0U) value -= initial_x;
            if (feature == 2U) value -= initial_z;
            global[frame * global_root_width + feature] = normalize_feature(model, value, feature);
        }
        for (std::uint32_t feature = 0; feature < local_root_width; ++feature)
            local[frame * local_root_width + feature] = normalize_feature(
                model, constraints.local_root[frame * local_root_width + feature], 5U + feature);

        poses[frame * internal_pose_width] = normalize_feature(
            model, constraints.global_root[frame * global_root_width + 1U], 8U);
        for (std::uint32_t feature = 0; feature < external_pose_width; ++feature)
            poses[frame * internal_pose_width + 1U + feature] = normalize_feature(
                model, constraints.poses[frame * external_pose_width + feature], 9U + feature);
    }
    if (trace != nullptr) {
        trace->normalized_global_root = global;
        trace->normalized_local_root = local;
        trace->normalized_poses = poses;
    }

    root_result root;
    auto stage_start = std::chrono::steady_clock::now();
    const auto finish_stage = [&](std::size_t index) {
        const auto now = std::chrono::steady_clock::now();
        if (trace) trace->stage_ms[index] = std::chrono::duration<double,std::milli>(now-stage_start).count();
        stage_start = now;
    };
    auto status = run_root_planner_auto_probe(*model.runtime, global, constraints.has_global_root,
        local, constraints.has_local_root, poses, constraints.has_poses, 6U, root, reason);
    if (status != MB_OK) return status;
    std::uint32_t tokens = 0U;
    float best = -std::numeric_limits<float>::infinity();
    for (std::uint32_t index = 0; index < 11U; ++index) {
        if (constraints.allowed_tokens[index] != 0U && root.duration_logits[index] > best) {
            best = root.duration_logits[index];
            tokens = 6U + index;
        }
    }
    if (tokens == 0U) {
        reason = "style disallows every reachable duration";
        return MB_INVALID_ARGUMENT;
    }
    if (tokens != 6U) {
        status = run_root_planner_auto_probe(*model.runtime, global, constraints.has_global_root,
            local, constraints.has_local_root, poses, constraints.has_poses, tokens, root, reason);
        if (status != MB_OK) return status;
    }
    const std::uint32_t frames = tokens * 4U;
    finish_stage(0U);
    if (trace != nullptr) {
        trace->duration_logits = root.duration_logits;
        trace->selected_tokens = tokens;
        trace->predicted_global_root = root.global_root_values;
    }

    std::vector<float> predicted_local(static_cast<std::size_t>(frames) * local_root_width);
    std::vector<float> raw_global(static_cast<std::size_t>(frames) * global_root_width);
    std::vector<float> heading(frames);
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        for (std::uint32_t feature = 0; feature < global_root_width; ++feature)
            raw_global[frame * global_root_width + feature] = unnormalize_feature(
                model, root.global_root_values[frame * global_root_width + feature], feature);
        heading[frame] = std::atan2(raw_global[frame * global_root_width + 4U],
                                    raw_global[frame * global_root_width + 3U]);
    }
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        const auto next = std::min(frame + 1U, frames - 1U);
        const auto velocity_from = frame + 1U < frames ? frame : frame - 1U;
        const auto velocity_to = frame + 1U < frames ? next : frame;
        const float angular_velocity = wrapped(heading[velocity_to] - heading[velocity_from]) * 30.0F;
        const float x_velocity = (raw_global[velocity_to * global_root_width] -
                                  raw_global[velocity_from * global_root_width]) * 30.0F;
        const float z_velocity = (raw_global[velocity_to * global_root_width + 2U] -
                                  raw_global[velocity_from * global_root_width + 2U]) * 30.0F;
        const float raw_values[4]{angular_velocity, x_velocity, z_velocity,
                                  raw_global[frame * global_root_width + 1U]};
        for (std::uint32_t feature = 0; feature < local_root_width; ++feature)
            predicted_local[frame * local_root_width + feature] =
                normalize_feature(model, raw_values[feature], 5U + feature);
    }
    if (constraints.has_local_root[7] != 0U)
        std::copy_n(local.data() + 7U * local_root_width, local_root_width,
                    predicted_local.data() + (frames - 1U) * local_root_width);
    if (trace != nullptr) trace->predicted_local_root = predicted_local;

    std::vector<float> pose_condition(static_cast<std::size_t>(frames) * internal_pose_width, 0.0F);
    std::vector<std::uint8_t> has_pose_condition(frames, 0U);
    for (std::uint32_t boundary = 0; boundary < 8U; ++boundary) {
        const std::uint32_t frame = boundary < 4U ? boundary : frames - 8U + boundary;
        std::copy_n(poses.data() + boundary * internal_pose_width, internal_pose_width,
                    pose_condition.data() + static_cast<std::size_t>(frame) * internal_pose_width);
        has_pose_condition[frame] = constraints.has_poses[boundary];
    }
    std::vector<float> pose_root(static_cast<std::size_t>(frames) * 4U);
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        const float * source = root.global_root_values.data() + frame * global_root_width;
        float * destination = pose_root.data() + frame * 4U;
        destination[0] = source[0]; destination[1] = source[2];
        destination[2] = source[3]; destination[3] = source[4];
    }
    if (trace != nullptr) {
        trace->pose_root_condition = pose_root;
        trace->pose_condition = pose_condition;
        trace->has_pose_condition = has_pose_condition;
    }
    std::vector<std::int32_t> pose_tokens(static_cast<std::size_t>(tokens) * 8U, 10);
    std::vector<float> logits;
    stage_start = std::chrono::steady_clock::now();
    status = run_pose_planner(*model.runtime, pose_tokens, pose_root, pose_condition,
                              has_pose_condition, tokens, tokens, logits, reason);
    if (status != MB_OK) return status;
    finish_stage(1U);
    std::vector<float> uniforms;
    if (!sampling_argmax) {
        uniforms.resize(logits.size());
        for (float & value : uniforms) value = sampling_uniform(seed);
    }
    if (!sample_pose_tokens(logits, uniforms, 10U, pose_tokens, reason)) return MB_INVALID_ARGUMENT;
    if (trace != nullptr) {
        trace->sampling_uniforms = uniforms;
        trace->pose_logits = logits;
        trace->pose_tokens = pose_tokens;
    }

    std::vector<float> codebook;
    if (!neural_copy_f32(*model.runtime, "vq-decoder", "quantizer.vq._codebook.embed",
                         codebook, reason)) return MB_INCOMPATIBLE_MODEL;
    if (codebook.size() != 8U * 10U * 32U) {
        reason = "VQ codebook shape mismatch";
        return MB_INCOMPATIBLE_MODEL;
    }
    std::vector<float> quantized(static_cast<std::size_t>(tokens) * 256U);
    for (std::uint32_t position = 0; position < tokens; ++position)
        for (std::uint32_t head = 0; head < 8U; ++head)
            for (std::uint32_t dimension = 0; dimension < 32U; ++dimension)
                    quantized[(head * 32U + dimension) * tokens + position] =
                    codebook[(head * 10U + static_cast<std::uint32_t>(
                        pose_tokens[position * 8U + head])) * 32U + dimension];
    if (trace != nullptr) trace->decoder_quantized = quantized;

    std::vector<float> external(static_cast<std::size_t>(frames) * 2U);
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        external[frame * 2U] = predicted_local[frame * local_root_width + 1U];
        external[frame * 2U + 1U] = predicted_local[frame * local_root_width + 2U];
    }
    if (trace != nullptr) trace->decoder_external_condition = external;
    // Upstream reuses pred_has_pose_cond for the decoder, including both the
    // four-frame source constraint and the dynamically placed four-frame end
    // constraint. Dropping the latter changes the entire decoded pose even
    // when root duration and pose-token selection agree.
    std::vector<std::uint8_t> decoder_mask(has_pose_condition.begin(),
                                           has_pose_condition.end());
    if (trace != nullptr) trace->decoder_target_mask = decoder_mask;
    std::vector<float> decoded;
    stage_start = std::chrono::steady_clock::now();
    status = run_vq_decoder(*model.runtime, quantized, external, pose_condition,
                            decoder_mask, tokens, decoded, reason);
    if (status != MB_OK) return status;
    finish_stage(2U);
    if (trace != nullptr) trace->decoder_output = decoded;
    status = decode_motion(model, decoded, frames, initial_x, initial_z,
                           initial_heading, output, reason);
    finish_stage(3U);
    if (status == MB_OK && selected_tokens != nullptr) *selected_tokens = tokens;
    return status;
}

} // namespace motionbricks::detail
