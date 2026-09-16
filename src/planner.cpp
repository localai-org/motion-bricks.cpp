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
    std::uint32_t tokens = 0U;
    const auto enabled_durations = std::count_if(
        constraints.allowed_tokens.begin(), constraints.allowed_tokens.end(),
        [](std::uint8_t value) { return value != 0U; });
    std::vector<float> duration_logits;
    auto status = MB_OK;
    if (enabled_durations == 1) {
        const auto found = std::find_if(constraints.allowed_tokens.begin(),
            constraints.allowed_tokens.end(), [](std::uint8_t value) { return value != 0U; });
        tokens = 6U + static_cast<std::uint32_t>(
            std::distance(constraints.allowed_tokens.begin(), found));
    } else if (enabled_durations > 1) {
        status = run_root_duration_probe(*model.runtime, global, constraints.has_global_root,
            local, constraints.has_local_root, poses, constraints.has_poses,
            duration_logits, reason);
        if (status != MB_OK) return status;
        float best = -std::numeric_limits<float>::infinity();
        for (std::uint32_t index = 0; index < 11U; ++index) {
            if (constraints.allowed_tokens[index] != 0U && duration_logits[index] > best) {
                best = duration_logits[index];
                tokens = 6U + index;
            }
        }
    }
    if (tokens == 0U) {
        reason = "style disallows every reachable duration";
        return MB_INVALID_ARGUMENT;
    }
    status = run_root_planner_auto_probe(*model.runtime, global, constraints.has_global_root,
        local, constraints.has_local_root, poses, constraints.has_poses, tokens, root, reason);
    if (status != MB_OK) return status;
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

    const auto & codebook = model.vq_codebook;
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

mb_status run_transition_batch(
    const mb_model & model,
    std::span<const transition_constraints> constraints,
    std::span<mb_motion *> outputs,
    std::span<const std::uint64_t> seeds,
    std::span<const std::uint8_t> sampling_argmax,
    std::span<std::uint32_t> selected_tokens,
    std::string & reason, bool force_native_batch) {
    const auto batch_size = constraints.size();
    if (!model.runtime || model.motion_mean.size() != 418U || model.motion_std.size() != 418U) {
        reason = "model is missing neural or normalization data";
        return MB_INCOMPATIBLE_MODEL;
    }
    if (batch_size == 0U || batch_size > 64U || outputs.size() != batch_size ||
        seeds.size() != batch_size || sampling_argmax.size() != batch_size ||
        (!selected_tokens.empty() && selected_tokens.size() != batch_size) ||
        std::any_of(outputs.begin(), outputs.end(), [](const mb_motion * value) { return value == nullptr; })) {
        reason = "transition batch arrays must have the same size in the range 1..64";
        return MB_INVALID_ARGUMENT;
    }
    // GGML's Zen 4 F32 kernels already use small-matrix GEMM inside one plan.
    // Measured B=4 fused graphs add 7--10% overhead on one/two CPU cores, so
    // the public batch API chooses the faster execution while retaining one
    // model and atomic ownership semantics. Vulkan and profiling can use the
    // native fused graph.
    if (!force_native_batch && neural_device(*model.runtime) == MB_DEVICE_CPU) {
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            std::uint32_t tokens = 0U;
            const auto status = run_transition(model, constraints[batch], *outputs[batch],
                selected_tokens.empty() ? nullptr : &tokens, nullptr, reason, seeds[batch],
                sampling_argmax[batch] != 0U);
            if (status != MB_OK) return status;
            if (!selected_tokens.empty()) selected_tokens[batch] = tokens;
        }
        return MB_OK;
    }

    struct work_item {
        float initial_x{};
        float initial_z{};
        float initial_heading{};
        std::vector<float> global;
        std::vector<float> local;
        std::vector<float> poses;
        root_result root;
        std::uint32_t tokens{};
        std::vector<float> predicted_local;
        std::vector<float> pose_root;
        std::vector<float> pose_condition;
        std::vector<std::uint8_t> has_pose_condition;
        std::vector<std::int32_t> pose_tokens;
        std::vector<float> quantized;
        std::vector<float> external;
    };
    std::vector<work_item> work(batch_size);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const auto & source = constraints[batch];
        auto & item = work[batch];
        item.initial_x = source.global_root[0];
        item.initial_z = source.global_root[2];
        item.initial_heading = std::atan2(source.global_root[4], source.global_root[3]);
        item.global.resize(8U * global_root_width);
        item.local.resize(8U * local_root_width);
        item.poses.resize(8U * internal_pose_width);
        for (std::uint32_t frame = 0; frame < 8U; ++frame) {
            for (std::uint32_t feature = 0; feature < global_root_width; ++feature) {
                float value = source.global_root[frame * global_root_width + feature];
                if (feature == 0U) value -= item.initial_x;
                if (feature == 2U) value -= item.initial_z;
                item.global[frame * global_root_width + feature] =
                    normalize_feature(model, value, feature);
            }
            for (std::uint32_t feature = 0; feature < local_root_width; ++feature)
                item.local[frame * local_root_width + feature] = normalize_feature(
                    model, source.local_root[frame * local_root_width + feature], 5U + feature);
            item.poses[frame * internal_pose_width] = normalize_feature(
                model, source.global_root[frame * global_root_width + 1U], 8U);
            for (std::uint32_t feature = 0; feature < external_pose_width; ++feature)
                item.poses[frame * internal_pose_width + 1U + feature] = normalize_feature(
                    model, source.poses[frame * external_pose_width + feature], 9U + feature);
        }
    }

    const auto append = []<typename T>(std::vector<T> & destination, std::span<const T> source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    std::vector<float> all_global, all_local, all_poses;
    std::vector<std::uint8_t> all_global_mask, all_local_mask, all_pose_mask;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        append(all_global, std::span<const float>(work[batch].global));
        append(all_local, std::span<const float>(work[batch].local));
        append(all_poses, std::span<const float>(work[batch].poses));
        append(all_global_mask, std::span<const std::uint8_t>(constraints[batch].has_global_root));
        append(all_local_mask, std::span<const std::uint8_t>(constraints[batch].has_local_root));
        append(all_pose_mask, std::span<const std::uint8_t>(constraints[batch].has_poses));
    }
    std::vector<std::vector<float>> duration_logits;
    const bool needs_duration_probe = std::any_of(
        constraints.begin(), constraints.end(), [](const transition_constraints & item) {
            return std::count_if(item.allowed_tokens.begin(), item.allowed_tokens.end(),
                [](std::uint8_t value) { return value != 0U; }) > 1;
        });
    auto status = MB_OK;
    if (needs_duration_probe) {
        status = run_root_duration_probe_batch(*model.runtime, all_global, all_global_mask,
            all_local, all_local_mask, all_poses, all_pose_mask,
            static_cast<std::uint32_t>(batch_size), duration_logits, reason);
        if (status != MB_OK) return status;
    }
    std::array<std::vector<std::size_t>, 11> buckets;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const auto enabled = std::count_if(constraints[batch].allowed_tokens.begin(),
            constraints[batch].allowed_tokens.end(), [](std::uint8_t value) { return value != 0U; });
        if (enabled == 1) {
            const auto found = std::find_if(constraints[batch].allowed_tokens.begin(),
                constraints[batch].allowed_tokens.end(), [](std::uint8_t value) { return value != 0U; });
            work[batch].tokens = 6U + static_cast<std::uint32_t>(
                std::distance(constraints[batch].allowed_tokens.begin(), found));
        } else if (enabled > 1) {
            float best = -std::numeric_limits<float>::infinity();
            for (std::uint32_t index = 0; index < 11U; ++index) {
                if (constraints[batch].allowed_tokens[index] != 0U &&
                    duration_logits[batch][index] > best) {
                    best = duration_logits[batch][index];
                    work[batch].tokens = 6U + index;
                }
            }
        }
        if (work[batch].tokens == 0U) {
            reason = "style disallows every reachable duration";
            return MB_INVALID_ARGUMENT;
        }
        buckets[work[batch].tokens - 6U].push_back(batch);
        if (!selected_tokens.empty()) selected_tokens[batch] = work[batch].tokens;
    }

    for (std::uint32_t token_index = 0U; token_index < buckets.size(); ++token_index) {
        const auto & members = buckets[token_index];
        if (members.empty()) continue;
        std::vector<float> bucket_global, bucket_local, bucket_poses;
        std::vector<std::uint8_t> bucket_global_mask, bucket_local_mask, bucket_pose_mask;
        for (const auto batch : members) {
            append(bucket_global, std::span<const float>(work[batch].global));
            append(bucket_local, std::span<const float>(work[batch].local));
            append(bucket_poses, std::span<const float>(work[batch].poses));
            append(bucket_global_mask, std::span<const std::uint8_t>(constraints[batch].has_global_root));
            append(bucket_local_mask, std::span<const std::uint8_t>(constraints[batch].has_local_root));
            append(bucket_pose_mask, std::span<const std::uint8_t>(constraints[batch].has_poses));
        }
        std::vector<root_result> roots;
        status = run_root_planner_auto_probe_batch(*model.runtime, bucket_global,
            bucket_global_mask, bucket_local, bucket_local_mask, bucket_poses,
            bucket_pose_mask, token_index + 6U, static_cast<std::uint32_t>(members.size()),
            roots, reason);
        if (status != MB_OK) return status;
        for (std::size_t index = 0; index < members.size(); ++index)
            work[members[index]].root = std::move(roots[index]);
    }

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        auto & item = work[batch];
        const auto frames = item.tokens * 4U;
        item.predicted_local.resize(static_cast<std::size_t>(frames) * local_root_width);
        std::vector<float> raw_global(static_cast<std::size_t>(frames) * global_root_width);
        std::vector<float> heading(frames);
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            for (std::uint32_t feature = 0; feature < global_root_width; ++feature)
                raw_global[frame * global_root_width + feature] = unnormalize_feature(
                    model, item.root.global_root_values[frame * global_root_width + feature], feature);
            heading[frame] = std::atan2(raw_global[frame * global_root_width + 4U],
                                        raw_global[frame * global_root_width + 3U]);
        }
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            const auto next = std::min(frame + 1U, frames - 1U);
            const auto velocity_from = frame + 1U < frames ? frame : frame - 1U;
            const auto velocity_to = frame + 1U < frames ? next : frame;
            const float raw_values[4]{
                wrapped(heading[velocity_to] - heading[velocity_from]) * 30.0F,
                (raw_global[velocity_to * global_root_width] -
                 raw_global[velocity_from * global_root_width]) * 30.0F,
                (raw_global[velocity_to * global_root_width + 2U] -
                 raw_global[velocity_from * global_root_width + 2U]) * 30.0F,
                raw_global[frame * global_root_width + 1U]};
            for (std::uint32_t feature = 0; feature < local_root_width; ++feature)
                item.predicted_local[frame * local_root_width + feature] =
                    normalize_feature(model, raw_values[feature], 5U + feature);
        }
        if (constraints[batch].has_local_root[7] != 0U)
            std::copy_n(item.local.data() + 7U * local_root_width, local_root_width,
                        item.predicted_local.data() + (frames - 1U) * local_root_width);
        item.pose_condition.assign(static_cast<std::size_t>(frames) * internal_pose_width, 0.0F);
        item.has_pose_condition.assign(frames, 0U);
        for (std::uint32_t boundary = 0; boundary < 8U; ++boundary) {
            const std::uint32_t frame = boundary < 4U ? boundary : frames - 8U + boundary;
            std::copy_n(item.poses.data() + boundary * internal_pose_width, internal_pose_width,
                        item.pose_condition.data() + static_cast<std::size_t>(frame) * internal_pose_width);
            item.has_pose_condition[frame] = constraints[batch].has_poses[boundary];
        }
        item.pose_root.resize(static_cast<std::size_t>(frames) * 4U);
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            const float * source = item.root.global_root_values.data() + frame * global_root_width;
            float * destination = item.pose_root.data() + frame * 4U;
            destination[0] = source[0]; destination[1] = source[2];
            destination[2] = source[3]; destination[3] = source[4];
        }
        item.pose_tokens.assign(static_cast<std::size_t>(item.tokens) * 8U, 10);
    }

    for (std::uint32_t token_index = 0U; token_index < buckets.size(); ++token_index) {
        const auto & members = buckets[token_index];
        if (members.empty()) continue;
        const auto tokens = token_index + 6U;
        std::vector<std::int32_t> bucket_pose_tokens;
        std::vector<float> bucket_pose_root, bucket_pose_condition;
        std::vector<std::uint8_t> bucket_has_pose_condition;
        for (const auto batch : members) {
            append(bucket_pose_tokens, std::span<const std::int32_t>(work[batch].pose_tokens));
            append(bucket_pose_root, std::span<const float>(work[batch].pose_root));
            append(bucket_pose_condition, std::span<const float>(work[batch].pose_condition));
            append(bucket_has_pose_condition,
                   std::span<const std::uint8_t>(work[batch].has_pose_condition));
        }
        std::vector<std::vector<float>> logits;
        status = run_pose_planner_batch(*model.runtime, bucket_pose_tokens, bucket_pose_root,
            bucket_pose_condition, bucket_has_pose_condition, tokens, tokens,
            static_cast<std::uint32_t>(members.size()), logits, reason);
        if (status != MB_OK) return status;
        for (std::size_t index = 0; index < members.size(); ++index) {
            const auto batch = members[index];
            std::vector<float> uniforms;
            if (sampling_argmax[batch] == 0U) {
                uniforms.resize(logits[index].size());
                auto random_state = seeds[batch];
                for (float & value : uniforms) value = sampling_uniform(random_state);
            }
            if (!sample_pose_tokens(logits[index], uniforms, 10U, work[batch].pose_tokens, reason))
                return MB_INVALID_ARGUMENT;
        }
    }

    const auto & codebook = model.vq_codebook;
    if (codebook.size() != 8U * 10U * 32U) {
        reason = "VQ codebook shape mismatch";
        return MB_INCOMPATIBLE_MODEL;
    }
    for (auto & item : work) {
        const auto frames = item.tokens * 4U;
        item.quantized.resize(static_cast<std::size_t>(item.tokens) * 256U);
        for (std::uint32_t position = 0; position < item.tokens; ++position)
            for (std::uint32_t head = 0; head < 8U; ++head)
                for (std::uint32_t dimension = 0; dimension < 32U; ++dimension)
                    item.quantized[(head * 32U + dimension) * item.tokens + position] =
                        codebook[(head * 10U + static_cast<std::uint32_t>(
                            item.pose_tokens[position * 8U + head])) * 32U + dimension];
        item.external.resize(static_cast<std::size_t>(frames) * 2U);
        for (std::uint32_t frame = 0; frame < frames; ++frame) {
            item.external[frame * 2U] = item.predicted_local[frame * local_root_width + 1U];
            item.external[frame * 2U + 1U] = item.predicted_local[frame * local_root_width + 2U];
        }
    }
    for (std::uint32_t token_index = 0U; token_index < buckets.size(); ++token_index) {
        const auto & members = buckets[token_index];
        if (members.empty()) continue;
        const auto tokens = token_index + 6U;
        std::vector<float> bucket_quantized, bucket_external, bucket_condition;
        std::vector<std::uint8_t> bucket_mask;
        for (const auto batch : members) {
            append(bucket_quantized, std::span<const float>(work[batch].quantized));
            append(bucket_external, std::span<const float>(work[batch].external));
            append(bucket_condition, std::span<const float>(work[batch].pose_condition));
            append(bucket_mask, std::span<const std::uint8_t>(work[batch].has_pose_condition));
        }
        std::vector<std::vector<float>> decoded;
        status = run_vq_decoder_batch(*model.runtime, bucket_quantized, bucket_external,
            bucket_condition, bucket_mask, tokens, static_cast<std::uint32_t>(members.size()),
            decoded, reason);
        if (status != MB_OK) return status;
        for (std::size_t index = 0; index < members.size(); ++index) {
            const auto batch = members[index];
            status = decode_motion(model, decoded[index], tokens * 4U,
                work[batch].initial_x, work[batch].initial_z, work[batch].initial_heading,
                *outputs[batch], reason);
            if (status != MB_OK) return status;
        }
    }
    return MB_OK;
}

} // namespace motionbricks::detail
