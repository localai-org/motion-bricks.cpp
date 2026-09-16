#include "root.hpp"

#include "neural_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml-backend.h>
#include <ggml.h>
#endif

namespace motionbricks::detail {
namespace {
#if defined(MOTIONBRICKS_HAVE_GGML)

struct context_deleter { void operator()(ggml_context * value) const noexcept { ggml_free(value); } };
struct buffer_deleter {
    void operator()(ggml_backend_buffer * value) const noexcept { ggml_backend_buffer_free(value); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

ggml_tensor * weight(const neural_runtime & runtime, const std::string & name,
                     std::string & reason) {
    auto * result = neural_weight(runtime, "root", name);
    if (result == nullptr && reason.empty()) reason = "missing root weight: " + name;
    return result;
}

ggml_tensor * linear(ggml_context * context, ggml_tensor * input,
                     ggml_tensor * matrix, ggml_tensor * bias) {
    return ggml_add(context, ggml_mul_mat(context, matrix, input), bias);
}

ggml_tensor * leaky_mlp(ggml_context * context, const neural_runtime & runtime,
                        ggml_tensor * input, const std::string & prefix,
                        std::string & reason) {
    auto * hidden = input;
    for (unsigned layer = 0; layer < 2U; ++layer) {
        const auto stem = prefix + ".fc_layers." + std::to_string(layer) + ".";
        hidden = ggml_leaky_relu(context, linear(context, hidden,
            weight(runtime, stem + "weight", reason), weight(runtime, stem + "bias", reason)),
            0.01F, false);
    }
    return linear(context, hidden, weight(runtime, prefix + ".forward_projection.weight", reason),
                  weight(runtime, prefix + ".forward_projection.bias", reason));
}

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input,
                         ggml_tensor * scale, ggml_tensor * bias) {
    return ggml_add(context, ggml_mul(context, ggml_norm(context, input, 1.0e-5F), scale), bias);
}

ggml_tensor * transformer(ggml_context * context, const neural_runtime & runtime,
                          ggml_tensor * input, const std::string & model_name,
                          unsigned layers, std::uint32_t sequence, std::string & reason) {
    constexpr std::int64_t embedding = 512, heads = 16, head_width = 32;
    auto * hidden = input;
    for (unsigned layer = 0; layer < layers; ++layer) {
        const auto prefix = model_name + ".layers." + std::to_string(layer) + ".";
        auto * qkv = linear(context, hidden,
            weight(runtime, prefix + "self_attn.in_proj_weight", reason),
            weight(runtime, prefix + "self_attn.in_proj_bias", reason));
        const auto stride = qkv->nb[1];
        auto * q_values = ggml_view_2d(context, qkv, embedding, sequence, stride, 0);
        auto * k_values = ggml_view_2d(context, qkv, embedding, sequence, stride,
                                      static_cast<std::size_t>(embedding) * sizeof(float));
        auto * v_values = ggml_view_2d(context, qkv, embedding, sequence, stride,
                                      static_cast<std::size_t>(embedding * 2) * sizeof(float));
        auto * query = ggml_permute(context,
            ggml_cont_3d(context, q_values, head_width, heads, sequence), 0, 2, 1, 3);
        auto * key = ggml_permute(context,
            ggml_cont_3d(context, k_values, head_width, heads, sequence), 0, 2, 1, 3);
        auto * scores = ggml_soft_max(context, ggml_scale(context,
            ggml_mul_mat(context, key, query), 1.0F / std::sqrt(static_cast<float>(head_width))));
        auto * value_transposed = ggml_cont_3d(context,
            ggml_permute(context,
                ggml_cont_3d(context, v_values, head_width, heads, sequence), 1, 2, 0, 3),
            sequence, head_width, heads);
        auto * attended = ggml_mul_mat(context, value_transposed, scores);
        auto * merged = ggml_cont_2d(context, ggml_permute(context, attended, 0, 2, 1, 3),
                                    embedding, sequence);
        auto * attention = linear(context, merged,
            weight(runtime, prefix + "self_attn.out_proj.weight", reason),
            weight(runtime, prefix + "self_attn.out_proj.bias", reason));
        hidden = layer_norm(context, ggml_add(context, hidden, attention),
            weight(runtime, prefix + "norm1.weight", reason),
            weight(runtime, prefix + "norm1.bias", reason));
        auto * feed = ggml_relu(context, linear(context, hidden,
            weight(runtime, prefix + "linear1.weight", reason),
            weight(runtime, prefix + "linear1.bias", reason)));
        feed = linear(context, feed, weight(runtime, prefix + "linear2.weight", reason),
                      weight(runtime, prefix + "linear2.bias", reason));
        hidden = layer_norm(context, ggml_add(context, hidden, feed),
            weight(runtime, prefix + "norm2.weight", reason),
            weight(runtime, prefix + "norm2.bias", reason));
    }
    return hidden;
}

ggml_tensor * conv(ggml_context * context, ggml_tensor * input,
                   ggml_tensor * kernel, ggml_tensor * bias, int padding, int dilation) {
    auto * columns = ggml_im2col(context, kernel, input, 1, 0, padding, 0,
                                dilation, 0, false, GGML_TYPE_F32);
    auto * output = ggml_mul_mat(context,
        ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
        ggml_reshape_2d(context, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    output = ggml_reshape_3d(context, output, columns->ne[1], kernel->ne[2], columns->ne[2]);
    return ggml_add(context, output, ggml_reshape_2d(context, bias, 1, bias->ne[0]));
}

ggml_tensor * residuals(ggml_context * context, const neural_runtime & runtime,
                        ggml_tensor * input, unsigned stage, std::string & reason) {
    static constexpr std::array dilations{27, 9, 3, 1};
    auto * hidden = input;
    for (unsigned block = 0; block < dilations.size(); ++block) {
        const auto prefix = "_conv_output.model." + std::to_string(stage) + ".0.model." +
                            std::to_string(block) + ".";
        auto * branch = conv(context, ggml_relu(context, hidden),
            weight(runtime, prefix + "conv1.weight", reason),
            weight(runtime, prefix + "conv1.bias", reason), dilations[block], dilations[block]);
        branch = conv(context, ggml_relu(context, branch),
            weight(runtime, prefix + "conv2.weight", reason),
            weight(runtime, prefix + "conv2.bias", reason), 0, 1);
        hidden = ggml_add(context, hidden, branch);
    }
    return hidden;
}

ggml_tensor * transpose_contiguous(ggml_context * context, ggml_tensor * input,
                                   std::int64_t first, std::int64_t second) {
    return ggml_cont_2d(context, ggml_transpose(context, input), first, second);
}

ggml_tensor * root_decoder(ggml_context * context, const neural_runtime & runtime,
                           ggml_tensor * tokens, ggml_tensor * external,
                           ggml_tensor * target, ggml_tensor * target_mask,
                           std::uint32_t positions, std::string & reason) {
    const auto frames = positions * 4U;
    auto * hidden = ggml_relu(context, conv(context, tokens,
        weight(runtime, "_conv_output.model.0.weight", reason),
        weight(runtime, "_conv_output.model.0.bias", reason), 1, 1));
    for (unsigned stage_index = 0; stage_index < 2U; ++stage_index) {
        const unsigned stage = stage_index + 2U;
        const auto group = 1U << (2U - stage_index);
        const auto stage_positions = positions * (1U << stage_index);
        auto * target_emb = ggml_relu(context, linear(context, target,
            weight(runtime, "_conv_output.target_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "_conv_output.target_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason)));
        auto * hidden_frames = transpose_contiguous(context, hidden, 512, stage_positions);
        hidden_frames = ggml_reshape_2d(context, hidden_frames, 512 / group, frames);
        hidden_frames = ggml_add(context, hidden_frames,
            ggml_mul(context, ggml_sub(context, target_emb, hidden_frames), target_mask));
        hidden_frames = ggml_reshape_2d(context, hidden_frames, 512, stage_positions);
        auto * external_grouped = ggml_reshape_2d(context, external, 512 * group, stage_positions);
        auto * fused = ggml_relu(context, linear(context,
            ggml_concat(context, hidden_frames, external_grouped, 0),
            weight(runtime, "_conv_output.external_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "_conv_output.external_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason)));
        hidden = transpose_contiguous(context, fused, stage_positions, 512);
        hidden = residuals(context, runtime, hidden, stage, reason);
        hidden = ggml_interpolate(context, hidden, hidden->ne[0] * 2, hidden->ne[1],
                                  hidden->ne[2], hidden->ne[3], GGML_SCALE_MODE_NEAREST);
        hidden = conv(context, hidden,
            weight(runtime, "_conv_output.model." + std::to_string(stage) + ".2.weight", reason),
            weight(runtime, "_conv_output.model." + std::to_string(stage) + ".2.bias", reason), 1, 1);
    }
    hidden = ggml_relu(context, conv(context, hidden,
        weight(runtime, "_conv_output.model.4.weight", reason),
        weight(runtime, "_conv_output.model.4.bias", reason), 1, 1));
    hidden = conv(context, hidden, weight(runtime, "_conv_output.model.6.weight", reason),
                  weight(runtime, "_conv_output.model.6.bias", reason), 1, 1);
    return transpose_contiguous(context, hidden, 5, frames);
}

ggml_tensor * concat_all(ggml_context * context,
                         const std::vector<ggml_tensor *> & tensors, int dimension) {
    auto * result = tensors.front();
    for (std::size_t index = 1; index < tensors.size(); ++index)
        result = ggml_concat(context, result, tensors[index], dimension);
    return result;
}

ggml_tensor * sequence_view(ggml_context * context, ggml_tensor * tensor,
                            std::int64_t width, std::uint32_t sequence,
                            std::uint32_t full_sequence, std::uint32_t batch,
                            std::uint32_t sequence_offset = 0U) {
    return ggml_view_2d(context, tensor, width, sequence, tensor->nb[1],
        static_cast<std::size_t>(batch * full_sequence + sequence_offset) * tensor->nb[1]);
}

ggml_tensor * transformer_batch(ggml_context * context, const neural_runtime & runtime,
                                ggml_tensor * input, const std::string & model_name,
                                unsigned layers, std::uint32_t sequence,
                                std::uint32_t batch_size, std::string & reason) {
    constexpr std::int64_t embedding = 512, heads = 16, head_width = 32;
    auto * hidden = input;
    for (unsigned layer = 0; layer < layers; ++layer) {
        const auto prefix = model_name + ".layers." + std::to_string(layer) + ".";
        auto * qkv = linear(context, hidden,
            weight(runtime, prefix + "self_attn.in_proj_weight", reason),
            weight(runtime, prefix + "self_attn.in_proj_bias", reason));
        std::vector<ggml_tensor *> merged_batches;
        merged_batches.reserve(batch_size);
        for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
            const auto base = static_cast<std::size_t>(batch * sequence) * qkv->nb[1];
            auto * q_values = ggml_view_2d(context, qkv, embedding, sequence,
                                           qkv->nb[1], base);
            auto * k_values = ggml_view_2d(context, qkv, embedding, sequence,
                                           qkv->nb[1], base + embedding * sizeof(float));
            auto * v_values = ggml_view_2d(context, qkv, embedding, sequence,
                                           qkv->nb[1], base + embedding * 2U * sizeof(float));
            auto * query = ggml_permute(context,
                ggml_cont_3d(context, q_values, head_width, heads, sequence), 0, 2, 1, 3);
            auto * key = ggml_permute(context,
                ggml_cont_3d(context, k_values, head_width, heads, sequence), 0, 2, 1, 3);
            auto * scores = ggml_soft_max(context, ggml_scale(context,
                ggml_mul_mat(context, key, query),
                1.0F / std::sqrt(static_cast<float>(head_width))));
            auto * value_transposed = ggml_cont_3d(context,
                ggml_permute(context,
                    ggml_cont_3d(context, v_values, head_width, heads, sequence), 1, 2, 0, 3),
                sequence, head_width, heads);
            auto * attended = ggml_mul_mat(context, value_transposed, scores);
            merged_batches.push_back(ggml_cont_2d(context,
                ggml_permute(context, attended, 0, 2, 1, 3), embedding, sequence));
        }
        auto * merged = concat_all(context, merged_batches, 1);
        auto * attention = linear(context, merged,
            weight(runtime, prefix + "self_attn.out_proj.weight", reason),
            weight(runtime, prefix + "self_attn.out_proj.bias", reason));
        hidden = layer_norm(context, ggml_add(context, hidden, attention),
            weight(runtime, prefix + "norm1.weight", reason),
            weight(runtime, prefix + "norm1.bias", reason));
        auto * feed = ggml_relu(context, linear(context, hidden,
            weight(runtime, prefix + "linear1.weight", reason),
            weight(runtime, prefix + "linear1.bias", reason)));
        feed = linear(context, feed, weight(runtime, prefix + "linear2.weight", reason),
                      weight(runtime, prefix + "linear2.bias", reason));
        hidden = layer_norm(context, ggml_add(context, hidden, feed),
            weight(runtime, prefix + "norm2.weight", reason),
            weight(runtime, prefix + "norm2.bias", reason));
    }
    return hidden;
}

ggml_tensor * transpose_contiguous_batch(ggml_context * context, ggml_tensor * input,
                                         std::int64_t first, std::int64_t second,
                                         std::uint32_t batch_size) {
    return ggml_cont_3d(context, ggml_transpose(context, input), first, second, batch_size);
}

ggml_tensor * conv_batch_independent(ggml_context * context, ggml_tensor * input,
                                     ggml_tensor * kernel, ggml_tensor * bias,
                                     int padding, int dilation,
                                     std::uint32_t batch_size) {
    std::vector<ggml_tensor *> batches;
    batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        auto * lane = ggml_view_2d(context, input, input->ne[0], input->ne[1],
                                   input->nb[1], batch * input->nb[2]);
        auto * result = conv(context, lane, kernel, bias, padding, dilation);
        batches.push_back(ggml_reshape_3d(context, result, result->ne[0], result->ne[1], 1));
    }
    return concat_all(context, batches, 2);
}

ggml_tensor * residuals_batch(ggml_context * context, const neural_runtime & runtime,
                              ggml_tensor * input, unsigned stage,
                              std::uint32_t batch_size, std::string & reason) {
    static constexpr std::array dilations{27, 9, 3, 1};
    auto * hidden = input;
    for (unsigned block = 0; block < dilations.size(); ++block) {
        const auto prefix = "_conv_output.model." + std::to_string(stage) + ".0.model." +
                            std::to_string(block) + ".";
        auto * branch = conv_batch_independent(context, ggml_relu(context, hidden),
            weight(runtime, prefix + "conv1.weight", reason),
            weight(runtime, prefix + "conv1.bias", reason), dilations[block], dilations[block],
            batch_size);
        branch = conv_batch_independent(context, ggml_relu(context, branch),
            weight(runtime, prefix + "conv2.weight", reason),
            weight(runtime, prefix + "conv2.bias", reason), 0, 1, batch_size);
        hidden = ggml_add(context, hidden, branch);
    }
    return hidden;
}

ggml_tensor * root_decoder_batch(ggml_context * context, const neural_runtime & runtime,
                                 ggml_tensor * tokens, ggml_tensor * external,
                                 ggml_tensor * target, ggml_tensor * target_mask,
                                 std::uint32_t positions, std::uint32_t batch_size,
                                 std::string & reason) {
    const auto frames = positions * 4U;
    auto * hidden = ggml_relu(context, conv_batch_independent(context, tokens,
        weight(runtime, "_conv_output.model.0.weight", reason),
        weight(runtime, "_conv_output.model.0.bias", reason), 1, 1, batch_size));
    for (unsigned stage_index = 0; stage_index < 2U; ++stage_index) {
        const unsigned stage = stage_index + 2U;
        const auto group = 1U << (2U - stage_index);
        const auto stage_positions = positions * (1U << stage_index);
        auto * target_emb = ggml_relu(context, linear(context, target,
            weight(runtime, "_conv_output.target_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "_conv_output.target_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason)));
        auto * hidden_frames = transpose_contiguous_batch(
            context, hidden, 512, stage_positions, batch_size);
        hidden_frames = ggml_reshape_3d(context, hidden_frames, 512 / group, frames, batch_size);
        hidden_frames = ggml_add(context, hidden_frames,
            ggml_mul(context, ggml_sub(context, target_emb, hidden_frames), target_mask));
        hidden_frames = ggml_reshape_3d(context, hidden_frames, 512, stage_positions, batch_size);
        auto * external_grouped = ggml_reshape_3d(
            context, external, 512 * group, stage_positions, batch_size);
        auto * fused = ggml_relu(context, linear(context,
            ggml_concat(context, hidden_frames, external_grouped, 0),
            weight(runtime, "_conv_output.external_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "_conv_output.external_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason)));
        hidden = transpose_contiguous_batch(
            context, fused, stage_positions, 512, batch_size);
        hidden = residuals_batch(context, runtime, hidden, stage, batch_size, reason);
        hidden = ggml_interpolate(context, hidden, hidden->ne[0] * 2, hidden->ne[1],
                                  hidden->ne[2], hidden->ne[3], GGML_SCALE_MODE_NEAREST);
        hidden = conv_batch_independent(context, hidden,
            weight(runtime, "_conv_output.model." + std::to_string(stage) + ".2.weight", reason),
            weight(runtime, "_conv_output.model." + std::to_string(stage) + ".2.bias", reason),
            1, 1, batch_size);
    }
    hidden = ggml_relu(context, conv_batch_independent(context, hidden,
        weight(runtime, "_conv_output.model.4.weight", reason),
        weight(runtime, "_conv_output.model.4.bias", reason), 1, 1, batch_size));
    hidden = conv_batch_independent(context, hidden,
        weight(runtime, "_conv_output.model.6.weight", reason),
        weight(runtime, "_conv_output.model.6.bias", reason), 1, 1, batch_size);
    return transpose_contiguous_batch(context, hidden, 5, frames, batch_size);
}

#endif
} // namespace

namespace {

mb_status run_root_planner_impl(const neural_runtime & runtime,
                           std::span<const float> global_root_values,
                           std::span<const std::uint8_t> has_global_root_values,
                           std::span<const float> local_root_values,
                           std::span<const std::uint8_t> has_local_root_values,
                           std::span<const float> poses,
                           std::span<const std::uint8_t> has_poses,
                           std::uint32_t requested_tokens,
                           std::uint32_t duration_input_tokens,
                           root_result & output,
                           std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)global_root_values; (void)has_global_root_values;
    (void)local_root_values; (void)has_local_root_values; (void)poses; (void)has_poses;
    (void)requested_tokens; (void)duration_input_tokens; (void)output;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t boundary_frames = 8U;
    if (requested_tokens < 6U || requested_tokens > 16U ||
        (duration_input_tokens != 18U &&
         (duration_input_tokens < 6U || duration_input_tokens > 16U)) ||
        global_root_values.size() != 40U ||
        local_root_values.size() != 32U || poses.size() != 2432U ||
        has_global_root_values.size() != boundary_frames ||
        has_local_root_values.size() != boundary_frames || has_poses.size() != boundary_frames) {
        reason = "root planner input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    const auto frames = requested_tokens * 4U;
    context_ptr context(ggml_init({48U * 1024U * 1024U, nullptr, true}));
    if (!context) { reason = "cannot allocate root graph metadata"; return MB_OUT_OF_MEMORY; }
    auto * global_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 5, boundary_frames);
    auto * local_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 4, boundary_frames);
    auto * pose_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 304, boundary_frames);
    auto * global_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, boundary_frames);
    auto * local_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, boundary_frames);
    auto * pose_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, boundary_frames);
    auto * dense_target = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 5, frames);
    auto * dense_target_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, frames);
    auto * duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
    auto * planned_duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
    for (auto * tensor : {global_input, local_input, pose_input, global_mask, local_mask, pose_mask,
                          dense_target, dense_target_mask, duration_input, planned_duration_input})
        ggml_set_input(tensor);

    auto blend = [&](ggml_tensor * projected, const char * absent_name, ggml_tensor * mask) {
        auto * absent = weight(runtime, absent_name, reason);
        return ggml_add(context.get(),
            ggml_mul(context.get(), ggml_sub(context.get(), projected, absent), mask), absent);
    };
    auto * pose_emb = blend(linear(context.get(), pose_input,
        weight(runtime, "_proj_local_pose.weight", reason), weight(runtime, "_proj_local_pose.bias", reason)),
        "_no_local_pose_emb", pose_mask);
    auto * local_emb = blend(linear(context.get(), local_input,
        weight(runtime, "_proj_local_root_value.weight", reason), weight(runtime, "_proj_local_root_value.bias", reason)),
        "_no_local_root_emb", local_mask);
    auto * global_emb = blend(linear(context.get(), global_input,
        weight(runtime, "_proj_global_root_value.weight", reason), weight(runtime, "_proj_global_root_value.bias", reason)),
        "_no_global_root_emb", global_mask);
    auto * boundary = ggml_concat(context.get(), ggml_concat(context.get(), pose_emb, local_emb, 0), global_emb, 0);
    auto * start = ggml_view_2d(context.get(), boundary, 384, 4, boundary->nb[1], 0);
    auto * end = ggml_view_2d(context.get(), boundary, 384, 4, boundary->nb[1], 4U * boundary->nb[1]);
    auto * start_frame = leaky_mlp(context.get(), runtime, start, "_proj_start_input", reason);
    auto * end_frame = leaky_mlp(context.get(), runtime, end, "_proj_end_input", reason);
    auto * frame_emb = ggml_concat(context.get(), start_frame, end_frame, 1);
    auto * positioned = ggml_add(context.get(), frame_emb, weight(runtime, "_input_position_emb.weight", reason));
    auto * duration_emb = ggml_get_rows(context.get(),
        weight(runtime, "_proj_input_num_tokens.weight", reason), duration_input);
    auto * first_input = ggml_concat(context.get(), duration_emb, positioned, 1);
    auto * first_output = transformer(context.get(), runtime, first_input,
        "_shared_transformer_model", 3, 9, reason);
    auto * first_token = ggml_view_2d(context.get(), first_output, 512, 1, first_output->nb[1], 0);
    auto * duration_logits = linear(context.get(), first_token,
        weight(runtime, "_proj_num_token_output_logit.weight", reason),
        weight(runtime, "_proj_num_token_output_logit.bias", reason));

    auto * middle_emb = ggml_get_rows(context.get(), weight(runtime, "_middle_token_emb.weight", reason),
                                      planned_duration_input);
    auto * positions_all = ggml_reshape_2d(context.get(), weight(runtime, "_position_emb.embed", reason), 512, 16);
    auto * positions_view = ggml_view_2d(context.get(), positions_all, 512, requested_tokens,
                                        positions_all->nb[1], 0);
    auto * second_input = ggml_concat(context.get(), middle_emb, positioned, 1);
    second_input = ggml_concat(context.get(), second_input, positions_view, 1);
    auto * second_output = transformer(context.get(), runtime, second_input,
        "_root_token_transformer_model", 3, 9U + requested_tokens, reason);
    auto * root_tokens = ggml_view_2d(context.get(), second_output, 512, requested_tokens,
                                     second_output->nb[1], 9U * second_output->nb[1]);
    auto * decoder_tokens = transpose_contiguous(context.get(), root_tokens, requested_tokens, 512);

    auto * has_frame = ggml_clamp(context.get(), ggml_add(context.get(),
        ggml_add(context.get(), global_mask, local_mask), pose_mask), 0.0F, 1.0F);
    auto * absent_frame = weight(runtime, "_conv_no_frame_emb", reason);
    auto * masked_frames = ggml_add(context.get(),
        ggml_mul(context.get(), ggml_sub(context.get(), frame_emb, absent_frame), has_frame), absent_frame);
    auto * first_frames = ggml_view_2d(context.get(), masked_frames, 512, 4, masked_frames->nb[1], 0);
    auto * last_frames = ggml_view_2d(context.get(), masked_frames, 512, 4, masked_frames->nb[1],
                                     4U * masked_frames->nb[1]);
    auto * middle_shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512, frames - 8U);
    auto * middle_frames = ggml_repeat(context.get(), absent_frame, middle_shape);
    auto * dense_frame = ggml_concat(context.get(), first_frames, middle_frames, 1);
    dense_frame = ggml_concat(context.get(), dense_frame, last_frames, 1);
    auto * root_values = root_decoder(context.get(), runtime, decoder_tokens, dense_frame,
                                     dense_target, dense_target_mask, requested_tokens, reason);
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    auto * graph = ggml_new_graph_custom(context.get(), 4096, false);
    ggml_build_forward_expand(graph, duration_logits);
    ggml_build_forward_expand(graph, root_values);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) { reason = "cannot allocate root compute buffer"; return MB_OUT_OF_MEMORY; }

    std::vector<float> global_mask_f(has_global_root_values.begin(), has_global_root_values.end());
    std::vector<float> local_mask_f(has_local_root_values.begin(), has_local_root_values.end());
    std::vector<float> pose_mask_f(has_poses.begin(), has_poses.end());
    std::vector<float> dense_values(static_cast<std::size_t>(frames) * 5U, 0.0F);
    std::vector<float> dense_mask_values(frames, 0.0F);
    for (std::uint32_t boundary_index = 0; boundary_index < 8U; ++boundary_index) {
        const auto frame = boundary_index < 4U ? boundary_index : frames - 8U + boundary_index;
        std::copy_n(global_root_values.data() + static_cast<std::size_t>(boundary_index) * 5U, 5,
                    dense_values.data() + static_cast<std::size_t>(frame) * 5U);
        dense_mask_values[frame] = has_global_root_values[boundary_index] != 0U ? 1.0F : 0.0F;
    }
    const std::int32_t duration_index = static_cast<std::int32_t>(duration_input_tokens - 6U);
    const std::int32_t planned_duration_index = static_cast<std::int32_t>(requested_tokens - 6U);
    ggml_backend_tensor_set(global_input, global_root_values.data(), 0, ggml_nbytes(global_input));
    ggml_backend_tensor_set(local_input, local_root_values.data(), 0, ggml_nbytes(local_input));
    ggml_backend_tensor_set(pose_input, poses.data(), 0, ggml_nbytes(pose_input));
    ggml_backend_tensor_set(global_mask, global_mask_f.data(), 0, ggml_nbytes(global_mask));
    ggml_backend_tensor_set(local_mask, local_mask_f.data(), 0, ggml_nbytes(local_mask));
    ggml_backend_tensor_set(pose_mask, pose_mask_f.data(), 0, ggml_nbytes(pose_mask));
    ggml_backend_tensor_set(dense_target, dense_values.data(), 0, ggml_nbytes(dense_target));
    ggml_backend_tensor_set(dense_target_mask, dense_mask_values.data(), 0, ggml_nbytes(dense_target_mask));
    ggml_backend_tensor_set(duration_input, &duration_index, 0, sizeof(duration_index));
    ggml_backend_tensor_set(planned_duration_input, &planned_duration_index, 0,
                            sizeof(planned_duration_index));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("root graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    output.tokens = requested_tokens;
    output.duration_logits.resize(12);
    output.global_root_values.resize(static_cast<std::size_t>(frames) * 5U);
    ggml_backend_tensor_get(duration_logits, output.duration_logits.data(), 0,
                            output.duration_logits.size() * sizeof(float));
    ggml_backend_tensor_get(root_values, output.global_root_values.data(), 0,
                            output.global_root_values.size() * sizeof(float));
    return MB_OK;
#endif
}

} // namespace

mb_status run_root_duration_probe_batch(
    const neural_runtime & runtime,
    std::span<const float> global_root_values,
    std::span<const std::uint8_t> has_global_root_values,
    std::span<const float> local_root_values,
    std::span<const std::uint8_t> has_local_root_values,
    std::span<const float> poses,
    std::span<const std::uint8_t> has_poses,
    std::uint32_t batch_size, std::vector<std::vector<float>> & duration_logits_output,
    std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)global_root_values; (void)has_global_root_values;
    (void)local_root_values; (void)has_local_root_values; (void)poses; (void)has_poses;
    (void)batch_size; (void)duration_logits_output;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t boundary_frames = 8U;
    if (batch_size == 0U || batch_size > 64U ||
        global_root_values.size() != static_cast<std::size_t>(batch_size) * 40U ||
        local_root_values.size() != static_cast<std::size_t>(batch_size) * 32U ||
        poses.size() != static_cast<std::size_t>(batch_size) * 2432U ||
        has_global_root_values.size() != static_cast<std::size_t>(batch_size) * boundary_frames ||
        has_local_root_values.size() != static_cast<std::size_t>(batch_size) * boundary_frames ||
        has_poses.size() != static_cast<std::size_t>(batch_size) * boundary_frames) {
        reason = "batched root duration input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    context_ptr context(ggml_init({64U * 1024U * 1024U, nullptr, true}));
    if (!context) { reason = "cannot allocate root duration graph metadata"; return MB_OUT_OF_MEMORY; }
    auto * global_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 5,
                                             boundary_frames * batch_size);
    auto * local_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 4,
                                            boundary_frames * batch_size);
    auto * pose_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 304,
                                           boundary_frames * batch_size);
    auto * global_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                            boundary_frames * batch_size);
    auto * local_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                           boundary_frames * batch_size);
    auto * pose_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                          boundary_frames * batch_size);
    auto * duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, batch_size);
    for (auto * tensor : {global_input, local_input, pose_input, global_mask, local_mask,
                          pose_mask, duration_input})
        ggml_set_input(tensor);
    auto blend = [&](ggml_tensor * projected, const char * absent_name, ggml_tensor * mask) {
        auto * absent = weight(runtime, absent_name, reason);
        return ggml_add(context.get(),
            ggml_mul(context.get(), ggml_sub(context.get(), projected, absent), mask), absent);
    };
    auto * pose_emb = blend(linear(context.get(), pose_input,
        weight(runtime, "_proj_local_pose.weight", reason), weight(runtime, "_proj_local_pose.bias", reason)),
        "_no_local_pose_emb", pose_mask);
    auto * local_emb = blend(linear(context.get(), local_input,
        weight(runtime, "_proj_local_root_value.weight", reason), weight(runtime, "_proj_local_root_value.bias", reason)),
        "_no_local_root_emb", local_mask);
    auto * global_emb = blend(linear(context.get(), global_input,
        weight(runtime, "_proj_global_root_value.weight", reason), weight(runtime, "_proj_global_root_value.bias", reason)),
        "_no_global_root_emb", global_mask);
    auto * boundary = ggml_concat(context.get(), ggml_concat(context.get(), pose_emb, local_emb, 0), global_emb, 0);
    std::vector<ggml_tensor *> starts, ends;
    starts.reserve(batch_size); ends.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        starts.push_back(sequence_view(context.get(), boundary, 384, 4, boundary_frames, batch));
        ends.push_back(sequence_view(context.get(), boundary, 384, 4, boundary_frames, batch, 4U));
    }
    auto * start_frame = leaky_mlp(context.get(), runtime,
        concat_all(context.get(), starts, 1), "_proj_start_input", reason);
    auto * end_frame = leaky_mlp(context.get(), runtime,
        concat_all(context.get(), ends, 1), "_proj_end_input", reason);
    std::vector<ggml_tensor *> frame_batches;
    frame_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        frame_batches.push_back(ggml_concat(context.get(),
            sequence_view(context.get(), start_frame, 512, 4, 4, batch),
            sequence_view(context.get(), end_frame, 512, 4, 4, batch), 1));
    auto * frame_emb = concat_all(context.get(), frame_batches, 1);
    auto * position_shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512,
                                               boundary_frames * batch_size);
    auto * positioned = ggml_add(context.get(), frame_emb,
        ggml_repeat(context.get(), weight(runtime, "_input_position_emb.weight", reason), position_shape));
    auto * duration_emb = ggml_get_rows(context.get(),
        weight(runtime, "_proj_input_num_tokens.weight", reason), duration_input);
    std::vector<ggml_tensor *> first_batches;
    first_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        first_batches.push_back(ggml_concat(context.get(),
            sequence_view(context.get(), duration_emb, 512, 1, 1, batch),
            sequence_view(context.get(), positioned, 512, boundary_frames,
                          boundary_frames, batch), 1));
    constexpr std::uint32_t first_sequence = 9U;
    auto * first_output = transformer_batch(context.get(), runtime,
        concat_all(context.get(), first_batches, 1), "_shared_transformer_model", 3,
        first_sequence, batch_size, reason);
    std::vector<ggml_tensor *> first_tokens;
    first_tokens.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        first_tokens.push_back(sequence_view(context.get(), first_output, 512, 1,
                                             first_sequence, batch));
    auto * duration_logits = linear(context.get(), concat_all(context.get(), first_tokens, 1),
        weight(runtime, "_proj_num_token_output_logit.weight", reason),
        weight(runtime, "_proj_num_token_output_logit.bias", reason));
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    auto * graph = ggml_new_graph_custom(context.get(), 4096U, false);
    ggml_build_forward_expand(graph, duration_logits);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) { reason = "cannot allocate root duration compute buffer"; return MB_OUT_OF_MEMORY; }
    std::vector<float> global_mask_f(has_global_root_values.begin(), has_global_root_values.end());
    std::vector<float> local_mask_f(has_local_root_values.begin(), has_local_root_values.end());
    std::vector<float> pose_mask_f(has_poses.begin(), has_poses.end());
    std::vector<std::int32_t> duration_indices(batch_size, 12);
    ggml_backend_tensor_set(global_input, global_root_values.data(), 0, ggml_nbytes(global_input));
    ggml_backend_tensor_set(local_input, local_root_values.data(), 0, ggml_nbytes(local_input));
    ggml_backend_tensor_set(pose_input, poses.data(), 0, ggml_nbytes(pose_input));
    ggml_backend_tensor_set(global_mask, global_mask_f.data(), 0, ggml_nbytes(global_mask));
    ggml_backend_tensor_set(local_mask, local_mask_f.data(), 0, ggml_nbytes(local_mask));
    ggml_backend_tensor_set(pose_mask, pose_mask_f.data(), 0, ggml_nbytes(pose_mask));
    ggml_backend_tensor_set(duration_input, duration_indices.data(), 0, ggml_nbytes(duration_input));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("root duration graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    std::vector<float> all_logits(static_cast<std::size_t>(batch_size) * 12U);
    ggml_backend_tensor_get(duration_logits, all_logits.data(), 0,
                            all_logits.size() * sizeof(float));
    duration_logits_output.assign(batch_size, {});
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        duration_logits_output[batch].assign(
            all_logits.begin() + static_cast<std::ptrdiff_t>(batch * 12U),
            all_logits.begin() + static_cast<std::ptrdiff_t>((batch + 1U) * 12U));
    return MB_OK;
#endif
}

mb_status run_root_duration_probe(
    const neural_runtime & runtime,
    std::span<const float> global_root_values,
    std::span<const std::uint8_t> has_global_root_values,
    std::span<const float> local_root_values,
    std::span<const std::uint8_t> has_local_root_values,
    std::span<const float> poses,
    std::span<const std::uint8_t> has_poses,
    std::vector<float> & duration_logits, std::string & reason) {
    std::vector<std::vector<float>> batch_logits;
    const auto status = run_root_duration_probe_batch(runtime, global_root_values,
        has_global_root_values, local_root_values, has_local_root_values, poses, has_poses,
        1U, batch_logits, reason);
    if (status == MB_OK) duration_logits = std::move(batch_logits.front());
    return status;
}

mb_status run_root_planner(const neural_runtime & runtime,
                           std::span<const float> global_root_values,
                           std::span<const std::uint8_t> has_global_root_values,
                           std::span<const float> local_root_values,
                           std::span<const std::uint8_t> has_local_root_values,
                           std::span<const float> poses,
                           std::span<const std::uint8_t> has_poses,
                           std::uint32_t requested_tokens,
                           root_result & output,
                           std::string & reason) {
    return run_root_planner_impl(runtime, global_root_values, has_global_root_values,
        local_root_values, has_local_root_values, poses, has_poses,
        requested_tokens, requested_tokens, output, reason);
}

mb_status run_root_planner_auto_probe(const neural_runtime & runtime,
                                      std::span<const float> global_root_values,
                                      std::span<const std::uint8_t> has_global_root_values,
                                      std::span<const float> local_root_values,
                                      std::span<const std::uint8_t> has_local_root_values,
                                      std::span<const float> poses,
                                      std::span<const std::uint8_t> has_poses,
                                      std::uint32_t candidate_tokens,
                                      root_result & output,
                                      std::string & reason) {
    return run_root_planner_impl(runtime, global_root_values, has_global_root_values,
        local_root_values, has_local_root_values, poses, has_poses,
        candidate_tokens, 18U, output, reason);
}

mb_status run_root_planner_auto_probe_batch(
    const neural_runtime & runtime,
    std::span<const float> global_root_values,
    std::span<const std::uint8_t> has_global_root_values,
    std::span<const float> local_root_values,
    std::span<const std::uint8_t> has_local_root_values,
    std::span<const float> poses,
    std::span<const std::uint8_t> has_poses,
    std::uint32_t candidate_tokens, std::uint32_t batch_size,
    std::vector<root_result> & output, std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)global_root_values; (void)has_global_root_values;
    (void)local_root_values; (void)has_local_root_values; (void)poses; (void)has_poses;
    (void)candidate_tokens; (void)batch_size; (void)output;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t boundary_frames = 8U;
    if (batch_size == 0U || batch_size > 64U ||
        candidate_tokens < 6U || candidate_tokens > 16U ||
        global_root_values.size() != static_cast<std::size_t>(batch_size) * 40U ||
        local_root_values.size() != static_cast<std::size_t>(batch_size) * 32U ||
        poses.size() != static_cast<std::size_t>(batch_size) * 2432U ||
        has_global_root_values.size() != static_cast<std::size_t>(batch_size) * boundary_frames ||
        has_local_root_values.size() != static_cast<std::size_t>(batch_size) * boundary_frames ||
        has_poses.size() != static_cast<std::size_t>(batch_size) * boundary_frames) {
        reason = "batched root planner input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    const auto frames = candidate_tokens * 4U;
    context_ptr context(ggml_init({128U * 1024U * 1024U, nullptr, true}));
    if (!context) { reason = "cannot allocate batched root graph metadata"; return MB_OUT_OF_MEMORY; }
    auto * global_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 5,
                                             boundary_frames * batch_size);
    auto * local_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 4,
                                            boundary_frames * batch_size);
    auto * pose_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 304,
                                           boundary_frames * batch_size);
    auto * global_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                            boundary_frames * batch_size);
    auto * local_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                           boundary_frames * batch_size);
    auto * pose_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                          boundary_frames * batch_size);
    auto * dense_target = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 5, frames, batch_size);
    auto * dense_target_mask = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, 1, frames, batch_size);
    auto * duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, batch_size);
    auto * planned_duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, batch_size);
    for (auto * tensor : {global_input, local_input, pose_input, global_mask, local_mask,
                          pose_mask, dense_target, dense_target_mask, duration_input,
                          planned_duration_input})
        ggml_set_input(tensor);

    auto blend = [&](ggml_tensor * projected, const char * absent_name, ggml_tensor * mask) {
        auto * absent = weight(runtime, absent_name, reason);
        return ggml_add(context.get(),
            ggml_mul(context.get(), ggml_sub(context.get(), projected, absent), mask), absent);
    };
    auto * pose_emb = blend(linear(context.get(), pose_input,
        weight(runtime, "_proj_local_pose.weight", reason), weight(runtime, "_proj_local_pose.bias", reason)),
        "_no_local_pose_emb", pose_mask);
    auto * local_emb = blend(linear(context.get(), local_input,
        weight(runtime, "_proj_local_root_value.weight", reason), weight(runtime, "_proj_local_root_value.bias", reason)),
        "_no_local_root_emb", local_mask);
    auto * global_emb = blend(linear(context.get(), global_input,
        weight(runtime, "_proj_global_root_value.weight", reason), weight(runtime, "_proj_global_root_value.bias", reason)),
        "_no_global_root_emb", global_mask);
    auto * boundary = ggml_concat(context.get(), ggml_concat(context.get(), pose_emb, local_emb, 0), global_emb, 0);
    std::vector<ggml_tensor *> starts, ends;
    starts.reserve(batch_size); ends.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        starts.push_back(sequence_view(context.get(), boundary, 384, 4, boundary_frames, batch));
        ends.push_back(sequence_view(context.get(), boundary, 384, 4, boundary_frames, batch, 4U));
    }
    auto * start_frame = leaky_mlp(context.get(), runtime,
        concat_all(context.get(), starts, 1), "_proj_start_input", reason);
    auto * end_frame = leaky_mlp(context.get(), runtime,
        concat_all(context.get(), ends, 1), "_proj_end_input", reason);
    std::vector<ggml_tensor *> frame_batches;
    frame_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        frame_batches.push_back(ggml_concat(context.get(),
            sequence_view(context.get(), start_frame, 512, 4, 4, batch),
            sequence_view(context.get(), end_frame, 512, 4, 4, batch), 1));
    }
    auto * frame_emb = concat_all(context.get(), frame_batches, 1);
    auto * position_shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512,
                                               boundary_frames * batch_size);
    auto * positioned = ggml_add(context.get(), frame_emb,
        ggml_repeat(context.get(), weight(runtime, "_input_position_emb.weight", reason), position_shape));
    auto * duration_emb = ggml_get_rows(context.get(),
        weight(runtime, "_proj_input_num_tokens.weight", reason), duration_input);
    std::vector<ggml_tensor *> first_batches;
    first_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        first_batches.push_back(ggml_concat(context.get(),
            sequence_view(context.get(), duration_emb, 512, 1, 1, batch),
            sequence_view(context.get(), positioned, 512, boundary_frames,
                          boundary_frames, batch), 1));
    }
    constexpr std::uint32_t first_sequence = 9U;
    auto * first_output = transformer_batch(context.get(), runtime,
        concat_all(context.get(), first_batches, 1), "_shared_transformer_model", 3,
        first_sequence, batch_size, reason);
    std::vector<ggml_tensor *> first_tokens;
    first_tokens.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        first_tokens.push_back(sequence_view(context.get(), first_output, 512, 1,
                                             first_sequence, batch));
    auto * duration_logits = linear(context.get(), concat_all(context.get(), first_tokens, 1),
        weight(runtime, "_proj_num_token_output_logit.weight", reason),
        weight(runtime, "_proj_num_token_output_logit.bias", reason));

    auto * middle_emb = ggml_get_rows(context.get(),
        weight(runtime, "_middle_token_emb.weight", reason), planned_duration_input);
    auto * positions_all = ggml_reshape_2d(context.get(),
        weight(runtime, "_position_emb.embed", reason), 512, 16);
    auto * positions_view = ggml_view_2d(context.get(), positions_all, 512, candidate_tokens,
                                         positions_all->nb[1], 0);
    std::vector<ggml_tensor *> second_batches;
    second_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        auto * second = ggml_concat(context.get(),
            sequence_view(context.get(), middle_emb, 512, 1, 1, batch),
            sequence_view(context.get(), positioned, 512, boundary_frames,
                          boundary_frames, batch), 1);
        second_batches.push_back(ggml_concat(context.get(), second, positions_view, 1));
    }
    const auto second_sequence = 9U + candidate_tokens;
    auto * second_output = transformer_batch(context.get(), runtime,
        concat_all(context.get(), second_batches, 1), "_root_token_transformer_model", 3,
        second_sequence, batch_size, reason);
    std::vector<ggml_tensor *> decoder_token_batches;
    decoder_token_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        auto * root_tokens = sequence_view(context.get(), second_output, 512, candidate_tokens,
                                           second_sequence, batch, 9U);
        auto * transposed = ggml_cont_2d(context.get(), ggml_transpose(context.get(), root_tokens),
                                         candidate_tokens, 512);
        decoder_token_batches.push_back(ggml_reshape_3d(
            context.get(), transposed, candidate_tokens, 512, 1));
    }
    auto * decoder_tokens = concat_all(context.get(), decoder_token_batches, 2);

    auto * has_frame = ggml_clamp(context.get(), ggml_add(context.get(),
        ggml_add(context.get(), global_mask, local_mask), pose_mask), 0.0F, 1.0F);
    auto * absent_frame = weight(runtime, "_conv_no_frame_emb", reason);
    auto * masked_frames = ggml_add(context.get(),
        ggml_mul(context.get(), ggml_sub(context.get(), frame_emb, absent_frame), has_frame), absent_frame);
    std::vector<ggml_tensor *> dense_frame_batches;
    dense_frame_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        auto * first_frames = sequence_view(context.get(), masked_frames, 512, 4,
                                            boundary_frames, batch);
        auto * last_frames = sequence_view(context.get(), masked_frames, 512, 4,
                                           boundary_frames, batch, 4U);
        auto * middle_shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 512, frames - 8U);
        auto * middle_frames = ggml_repeat(context.get(), absent_frame, middle_shape);
        auto * dense = ggml_concat(context.get(), first_frames, middle_frames, 1);
        dense = ggml_concat(context.get(), dense, last_frames, 1);
        dense_frame_batches.push_back(ggml_reshape_3d(context.get(), dense, 512, frames, 1));
    }
    auto * dense_frame = concat_all(context.get(), dense_frame_batches, 2);
    auto * root_values = root_decoder_batch(context.get(), runtime, decoder_tokens, dense_frame,
        dense_target, dense_target_mask, candidate_tokens, batch_size, reason);
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    auto * graph = ggml_new_graph_custom(context.get(), 8192, false);
    ggml_build_forward_expand(graph, duration_logits);
    ggml_build_forward_expand(graph, root_values);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) { reason = "cannot allocate batched root compute buffer"; return MB_OUT_OF_MEMORY; }

    std::vector<float> global_mask_f(has_global_root_values.begin(), has_global_root_values.end());
    std::vector<float> local_mask_f(has_local_root_values.begin(), has_local_root_values.end());
    std::vector<float> pose_mask_f(has_poses.begin(), has_poses.end());
    std::vector<float> dense_values(static_cast<std::size_t>(batch_size) * frames * 5U, 0.0F);
    std::vector<float> dense_mask_values(static_cast<std::size_t>(batch_size) * frames, 0.0F);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        for (std::uint32_t boundary_index = 0; boundary_index < 8U; ++boundary_index) {
            const auto frame = boundary_index < 4U ? boundary_index : frames - 8U + boundary_index;
            const auto source = static_cast<std::size_t>(batch) * 40U + boundary_index * 5U;
            const auto destination = (static_cast<std::size_t>(batch) * frames + frame) * 5U;
            std::copy_n(global_root_values.data() + source, 5, dense_values.data() + destination);
            dense_mask_values[static_cast<std::size_t>(batch) * frames + frame] =
                has_global_root_values[static_cast<std::size_t>(batch) * 8U + boundary_index] != 0U
                    ? 1.0F : 0.0F;
        }
    }
    std::vector<std::int32_t> duration_indices(batch_size, 12);
    std::vector<std::int32_t> planned_duration_indices(
        batch_size, static_cast<std::int32_t>(candidate_tokens - 6U));
    ggml_backend_tensor_set(global_input, global_root_values.data(), 0, ggml_nbytes(global_input));
    ggml_backend_tensor_set(local_input, local_root_values.data(), 0, ggml_nbytes(local_input));
    ggml_backend_tensor_set(pose_input, poses.data(), 0, ggml_nbytes(pose_input));
    ggml_backend_tensor_set(global_mask, global_mask_f.data(), 0, ggml_nbytes(global_mask));
    ggml_backend_tensor_set(local_mask, local_mask_f.data(), 0, ggml_nbytes(local_mask));
    ggml_backend_tensor_set(pose_mask, pose_mask_f.data(), 0, ggml_nbytes(pose_mask));
    ggml_backend_tensor_set(dense_target, dense_values.data(), 0, ggml_nbytes(dense_target));
    ggml_backend_tensor_set(dense_target_mask, dense_mask_values.data(), 0,
                            ggml_nbytes(dense_target_mask));
    ggml_backend_tensor_set(duration_input, duration_indices.data(), 0, ggml_nbytes(duration_input));
    ggml_backend_tensor_set(planned_duration_input, planned_duration_indices.data(), 0,
                            ggml_nbytes(planned_duration_input));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("batched root graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    std::vector<float> all_duration_logits(static_cast<std::size_t>(batch_size) * 12U);
    std::vector<float> all_root_values(static_cast<std::size_t>(batch_size) * frames * 5U);
    ggml_backend_tensor_get(duration_logits, all_duration_logits.data(), 0,
                            all_duration_logits.size() * sizeof(float));
    ggml_backend_tensor_get(root_values, all_root_values.data(), 0,
                            all_root_values.size() * sizeof(float));
    output.assign(batch_size, {});
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        output[batch].tokens = candidate_tokens;
        output[batch].duration_logits.assign(
            all_duration_logits.begin() + static_cast<std::ptrdiff_t>(batch * 12U),
            all_duration_logits.begin() + static_cast<std::ptrdiff_t>((batch + 1U) * 12U));
        output[batch].global_root_values.assign(
            all_root_values.begin() + static_cast<std::ptrdiff_t>(batch * frames * 5U),
            all_root_values.begin() + static_cast<std::ptrdiff_t>((batch + 1U) * frames * 5U));
    }
    return MB_OK;
#endif
}

} // namespace motionbricks::detail
