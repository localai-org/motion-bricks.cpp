#include "pose.hpp"

#include "neural_runtime.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml-backend.h>
#include <ggml.h>
#endif

namespace motionbricks::detail {
namespace {

#if defined(MOTIONBRICKS_HAVE_GGML)

struct context_deleter {
    void operator()(ggml_context * value) const noexcept { ggml_free(value); }
};
struct buffer_deleter {
    void operator()(ggml_backend_buffer * value) const noexcept { ggml_backend_buffer_free(value); }
};
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

ggml_tensor * weight(const neural_runtime & runtime, const std::string & name,
                     std::string & reason) {
    auto * result = neural_weight(runtime, "pose", name);
    if (result == nullptr && reason.empty()) reason = "missing pose weight: " + name;
    return result;
}

ggml_tensor * linear(ggml_context * context, ggml_tensor * input,
                     ggml_tensor * matrix, ggml_tensor * bias) {
    return ggml_add(context, ggml_mul_mat(context, matrix, input), bias);
}

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input,
                         ggml_tensor * scale, ggml_tensor * bias) {
    return ggml_add(context, ggml_mul(context, ggml_norm(context, input, 1.0e-5F), scale), bias);
}

ggml_tensor * transformer_layer(ggml_context * context, const neural_runtime & runtime,
                                ggml_tensor * input, unsigned layer, std::uint32_t positions,
                                ggml_tensor * attention_mask,
                                std::string & reason) {
    constexpr std::int64_t embedding = 1024;
    constexpr std::int64_t heads = 16;
    constexpr std::int64_t head_width = embedding / heads;
    const auto prefix = "_transformer_model.layers." + std::to_string(layer) + ".";
    auto * qkv_weight = weight(runtime, prefix + "self_attn.in_proj_weight", reason);
    auto * qkv_bias = weight(runtime, prefix + "self_attn.in_proj_bias", reason);
    auto * out_weight = weight(runtime, prefix + "self_attn.out_proj.weight", reason);
    auto * out_bias = weight(runtime, prefix + "self_attn.out_proj.bias", reason);
    auto * norm1_weight = weight(runtime, prefix + "norm1.weight", reason);
    auto * norm1_bias = weight(runtime, prefix + "norm1.bias", reason);
    auto * linear1_weight = weight(runtime, prefix + "linear1.weight", reason);
    auto * linear1_bias = weight(runtime, prefix + "linear1.bias", reason);
    auto * linear2_weight = weight(runtime, prefix + "linear2.weight", reason);
    auto * linear2_bias = weight(runtime, prefix + "linear2.bias", reason);
    auto * norm2_weight = weight(runtime, prefix + "norm2.weight", reason);
    auto * norm2_bias = weight(runtime, prefix + "norm2.bias", reason);
    if (!reason.empty()) return nullptr;

    auto * qkv = linear(context, input, qkv_weight, qkv_bias);
    const auto stride = qkv->nb[1];
    auto * query_values = ggml_view_2d(context, qkv, embedding, positions, stride, 0);
    auto * key_values = ggml_view_2d(context, qkv, embedding, positions, stride,
                                    static_cast<std::size_t>(embedding) * sizeof(float));
    auto * value_values = ggml_view_2d(context, qkv, embedding, positions, stride,
                                      static_cast<std::size_t>(embedding * 2) * sizeof(float));
    auto * query = ggml_permute(context,
        ggml_cont_3d(context, query_values, head_width, heads, positions), 0, 2, 1, 3);
    auto * key = ggml_permute(context,
        ggml_cont_3d(context, key_values, head_width, heads, positions), 0, 2, 1, 3);
    auto * scores = ggml_soft_max_ext(context, ggml_mul_mat(context, key, query),
        attention_mask, 1.0F / std::sqrt(static_cast<float>(head_width)), 0.0F);
    auto * value_transposed = ggml_cont_3d(context,
        ggml_permute(context,
            ggml_cont_3d(context, value_values, head_width, heads, positions), 1, 2, 0, 3),
        positions, head_width, heads);
    auto * attended = ggml_mul_mat(context, value_transposed, scores);
    auto * merged = ggml_cont_2d(context, ggml_permute(context, attended, 0, 2, 1, 3),
                                embedding, positions);
    auto * attention_output = linear(context, merged, out_weight, out_bias);
    auto * normalized_attention = layer_norm(
        context, ggml_add(context, input, attention_output), norm1_weight, norm1_bias);
    auto * feed_forward = ggml_relu(context,
        linear(context, normalized_attention, linear1_weight, linear1_bias));
    feed_forward = linear(context, feed_forward, linear2_weight, linear2_bias);
    return layer_norm(context, ggml_add(context, normalized_attention, feed_forward),
                      norm2_weight, norm2_bias);
}

ggml_tensor * concat_all(ggml_context * context,
                         const std::vector<ggml_tensor *> & tensors, int dimension) {
    auto * result = tensors.front();
    for (std::size_t index = 1; index < tensors.size(); ++index)
        result = ggml_concat(context, result, tensors[index], dimension);
    return result;
}

ggml_tensor * transformer_layer_batch(
    ggml_context * context, const neural_runtime & runtime, ggml_tensor * input,
    unsigned layer, std::uint32_t positions, std::uint32_t batch_size,
    ggml_tensor * attention_mask, std::string & reason) {
    constexpr std::int64_t embedding = 1024;
    constexpr std::int64_t heads = 16;
    constexpr std::int64_t head_width = embedding / heads;
    const auto prefix = "_transformer_model.layers." + std::to_string(layer) + ".";
    auto * qkv_weight = weight(runtime, prefix + "self_attn.in_proj_weight", reason);
    auto * qkv_bias = weight(runtime, prefix + "self_attn.in_proj_bias", reason);
    auto * out_weight = weight(runtime, prefix + "self_attn.out_proj.weight", reason);
    auto * out_bias = weight(runtime, prefix + "self_attn.out_proj.bias", reason);
    auto * norm1_weight = weight(runtime, prefix + "norm1.weight", reason);
    auto * norm1_bias = weight(runtime, prefix + "norm1.bias", reason);
    auto * linear1_weight = weight(runtime, prefix + "linear1.weight", reason);
    auto * linear1_bias = weight(runtime, prefix + "linear1.bias", reason);
    auto * linear2_weight = weight(runtime, prefix + "linear2.weight", reason);
    auto * linear2_bias = weight(runtime, prefix + "linear2.bias", reason);
    auto * norm2_weight = weight(runtime, prefix + "norm2.weight", reason);
    auto * norm2_bias = weight(runtime, prefix + "norm2.bias", reason);
    if (!reason.empty()) return nullptr;

    auto * qkv = linear(context, input, qkv_weight, qkv_bias);
    std::vector<ggml_tensor *> merged_batches;
    merged_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        const auto base = static_cast<std::size_t>(batch * positions) * qkv->nb[1];
        auto * query_values = ggml_view_2d(context, qkv, embedding, positions,
                                           qkv->nb[1], base);
        auto * key_values = ggml_view_2d(context, qkv, embedding, positions,
                                         qkv->nb[1], base + embedding * sizeof(float));
        auto * value_values = ggml_view_2d(context, qkv, embedding, positions,
                                           qkv->nb[1], base + embedding * 2U * sizeof(float));
        auto * query = ggml_permute(context,
            ggml_cont_3d(context, query_values, head_width, heads, positions), 0, 2, 1, 3);
        auto * key = ggml_permute(context,
            ggml_cont_3d(context, key_values, head_width, heads, positions), 0, 2, 1, 3);
        auto * scores = ggml_soft_max_ext(context, ggml_mul_mat(context, key, query),
            attention_mask, 1.0F / std::sqrt(static_cast<float>(head_width)), 0.0F);
        auto * value_transposed = ggml_cont_3d(context,
            ggml_permute(context,
                ggml_cont_3d(context, value_values, head_width, heads, positions), 1, 2, 0, 3),
            positions, head_width, heads);
        auto * attended = ggml_mul_mat(context, value_transposed, scores);
        merged_batches.push_back(ggml_cont_2d(context,
            ggml_permute(context, attended, 0, 2, 1, 3), embedding, positions));
    }
    auto * attention_output = linear(context, concat_all(context, merged_batches, 1),
                                     out_weight, out_bias);
    auto * normalized_attention = layer_norm(
        context, ggml_add(context, input, attention_output), norm1_weight, norm1_bias);
    auto * feed_forward = ggml_relu(context,
        linear(context, normalized_attention, linear1_weight, linear1_bias));
    feed_forward = linear(context, feed_forward, linear2_weight, linear2_bias);
    return layer_norm(context, ggml_add(context, normalized_attention, feed_forward),
                      norm2_weight, norm2_bias);
}

#endif

} // namespace

mb_status run_pose_planner(const neural_runtime & runtime,
                           std::span<const std::int32_t> pose_tokens,
                           std::span<const float> local_root_values,
                           std::span<const float> pose_condition,
                           std::span<const std::uint8_t> has_pose_condition,
                           std::uint32_t positions,
                           std::uint32_t num_tokens,
                           std::vector<float> & logits,
                           std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)pose_tokens; (void)local_root_values; (void)pose_condition;
    (void)has_pose_condition; (void)positions; (void)num_tokens; (void)logits;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t pose_heads = 8;
    constexpr std::uint32_t frames_per_position = 4;
    constexpr std::uint32_t root_width = 4;
    constexpr std::uint32_t pose_width = 304;
    if (positions < 6U || positions > 16U || num_tokens < 6U || num_tokens > positions) {
        reason = "pose planner token count is outside the released model range";
        return MB_INVALID_ARGUMENT;
    }
    const auto frames = positions * frames_per_position;
    if (pose_tokens.size() != static_cast<std::size_t>(positions) * pose_heads ||
        local_root_values.size() != static_cast<std::size_t>(frames) * root_width ||
        pose_condition.size() != static_cast<std::size_t>(frames) * pose_width ||
        has_pose_condition.size() != frames) {
        reason = "pose planner input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }

    constexpr std::size_t context_bytes = 32U * 1024U * 1024U;
    context_ptr context(ggml_init({context_bytes, nullptr, true}));
    if (!context) {
        reason = "cannot allocate pose graph metadata";
        return MB_OUT_OF_MEMORY;
    }
    auto * token_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32,
                                            static_cast<std::int64_t>(pose_heads) * positions);
    auto * root_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, root_width, frames);
    auto * condition_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, pose_width, frames);
    auto * mask_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, frames);
    auto * duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, 1);
    auto * attention_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32,
                                               positions, positions);
    ggml_set_input(token_input);
    ggml_set_input(root_input);
    ggml_set_input(condition_input);
    ggml_set_input(mask_input);
    ggml_set_input(duration_input);
    ggml_set_input(attention_mask);

    auto * token_embedding_weight = weight(runtime, "_pose_token_emb.weight", reason);
    auto * token_embedding = ggml_get_rows(context.get(), token_embedding_weight, token_input);
    token_embedding = ggml_reshape_2d(context.get(), token_embedding, 256, positions);
    token_embedding = ggml_leaky_relu(context.get(), linear(context.get(), token_embedding,
        weight(runtime, "_proj_pose_token_emb.fc_layers.0.weight", reason),
        weight(runtime, "_proj_pose_token_emb.fc_layers.0.bias", reason)), 0.01F, false);
    token_embedding = ggml_leaky_relu(context.get(), linear(context.get(), token_embedding,
        weight(runtime, "_proj_pose_token_emb.fc_layers.1.weight", reason),
        weight(runtime, "_proj_pose_token_emb.fc_layers.1.bias", reason)), 0.01F, false);
    token_embedding = linear(context.get(), token_embedding,
        weight(runtime, "_proj_pose_token_emb.forward_projection.weight", reason),
        weight(runtime, "_proj_pose_token_emb.forward_projection.bias", reason));
    token_embedding = ggml_reshape_2d(context.get(), token_embedding, 160, frames);

    auto * condition_embedding = linear(context.get(), condition_input,
        weight(runtime, "_proj_local_pose.weight", reason),
        weight(runtime, "_proj_local_pose.bias", reason));
    auto * pose_embedding = ggml_add(context.get(), token_embedding,
        ggml_mul(context.get(), ggml_sub(context.get(), condition_embedding, token_embedding),
                 mask_input));
    pose_embedding = ggml_reshape_2d(context.get(), pose_embedding, 640, positions);

    auto * root_embedding = linear(context.get(),
        ggml_reshape_2d(context.get(), root_input, 16, positions),
        weight(runtime, "_proj_local_root_values.weight", reason),
        weight(runtime, "_proj_local_root_values.bias", reason));
    auto * duration_embedding = ggml_get_rows(context.get(),
        weight(runtime, "_proj_num_valid_positions.weight", reason), duration_input);
    duration_embedding = ggml_repeat(context.get(), duration_embedding,
        ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 128, positions));
    auto * combined = ggml_concat(context.get(), pose_embedding, root_embedding, 0);
    combined = ggml_concat(context.get(), combined, duration_embedding, 0);
    auto * hidden = ggml_relu(context.get(), linear(context.get(), combined,
        weight(runtime, "_proj_input.0.weight", reason),
        weight(runtime, "_proj_input.0.bias", reason)));
    auto * position_weight = weight(runtime, "_position_emb.embed", reason);
    auto * position_matrix = ggml_reshape_2d(context.get(), position_weight, 1024, 16);
    auto * position_view = ggml_view_2d(context.get(), position_matrix, 1024, positions,
                                       position_matrix->nb[1], 0);
    hidden = ggml_add(context.get(), hidden, position_view);
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    for (unsigned layer = 0; layer < 16U; ++layer) {
        hidden = transformer_layer(context.get(), runtime, hidden, layer, positions,
                                   attention_mask, reason);
        if (hidden == nullptr) return MB_INCOMPATIBLE_MODEL;
    }
    auto * output = linear(context.get(), hidden,
        weight(runtime, "_proj_pose_output_logit.weight", reason),
        weight(runtime, "_proj_pose_output_logit.bias", reason));
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;

    auto * graph = ggml_new_graph_custom(context.get(), 2048, false);
    ggml_build_forward_expand(graph, output);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) {
        reason = "cannot allocate pose compute buffer";
        return MB_OUT_OF_MEMORY;
    }
    std::vector<std::int32_t> offset_tokens(pose_tokens.begin(), pose_tokens.end());
    for (std::uint32_t position = 0; position < positions; ++position)
        for (std::uint32_t head = 0; head < pose_heads; ++head)
            offset_tokens[static_cast<std::size_t>(position) * pose_heads + head] +=
                static_cast<std::int32_t>(head * 11U);
    std::vector<float> mask(has_pose_condition.begin(), has_pose_condition.end());
    std::vector<float> padding_mask(static_cast<std::size_t>(positions) * positions);
    for (std::uint32_t query = 0; query < positions; ++query)
        for (std::uint32_t key = num_tokens; key < positions; ++key)
            padding_mask[static_cast<std::size_t>(query) * positions + key] =
                -std::numeric_limits<float>::infinity();
    const std::int32_t duration_index = static_cast<std::int32_t>(num_tokens - 6U);
    ggml_backend_tensor_set(token_input, offset_tokens.data(), 0, ggml_nbytes(token_input));
    ggml_backend_tensor_set(root_input, local_root_values.data(), 0, ggml_nbytes(root_input));
    ggml_backend_tensor_set(condition_input, pose_condition.data(), 0, ggml_nbytes(condition_input));
    ggml_backend_tensor_set(mask_input, mask.data(), 0, ggml_nbytes(mask_input));
    ggml_backend_tensor_set(duration_input, &duration_index, 0, sizeof(duration_index));
    ggml_backend_tensor_set(attention_mask, padding_mask.data(), 0,
                            ggml_nbytes(attention_mask));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("pose graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    logits.resize(static_cast<std::size_t>(positions) * 80U);
    ggml_backend_tensor_get(output, logits.data(), 0, logits.size() * sizeof(float));
    return MB_OK;
#endif
}

mb_status run_pose_planner_batch(
    const neural_runtime & runtime,
    std::span<const std::int32_t> pose_tokens,
    std::span<const float> local_root_values,
    std::span<const float> pose_condition,
    std::span<const std::uint8_t> has_pose_condition,
    std::uint32_t positions, std::uint32_t num_tokens,
    std::uint32_t batch_size, std::vector<std::vector<float>> & logits,
    std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)pose_tokens; (void)local_root_values; (void)pose_condition;
    (void)has_pose_condition; (void)positions; (void)num_tokens; (void)batch_size;
    (void)logits;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t pose_heads = 8;
    constexpr std::uint32_t frames_per_position = 4;
    constexpr std::uint32_t root_width = 4;
    constexpr std::uint32_t pose_width = 304;
    if (batch_size == 0U || batch_size > 64U || positions < 6U || positions > 16U ||
        num_tokens < 6U || num_tokens > positions) {
        reason = "batched pose planner dimensions are outside the released model range";
        return MB_INVALID_ARGUMENT;
    }
    const auto frames = positions * frames_per_position;
    if (pose_tokens.size() != static_cast<std::size_t>(batch_size) * positions * pose_heads ||
        local_root_values.size() != static_cast<std::size_t>(batch_size) * frames * root_width ||
        pose_condition.size() != static_cast<std::size_t>(batch_size) * frames * pose_width ||
        has_pose_condition.size() != static_cast<std::size_t>(batch_size) * frames) {
        reason = "batched pose planner input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }

    context_ptr context(ggml_init({128U * 1024U * 1024U, nullptr, true}));
    if (!context) { reason = "cannot allocate batched pose graph metadata"; return MB_OUT_OF_MEMORY; }
    auto * token_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32,
        static_cast<std::int64_t>(batch_size) * pose_heads * positions);
    auto * root_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, root_width,
                                           frames * batch_size);
    auto * condition_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, pose_width,
                                                frames * batch_size);
    auto * mask_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1,
                                           frames * batch_size);
    auto * duration_input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, batch_size);
    auto * attention_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32,
                                               positions, positions);
    for (auto * tensor : {token_input, root_input, condition_input, mask_input,
                          duration_input, attention_mask})
        ggml_set_input(tensor);

    auto * token_embedding = ggml_get_rows(context.get(),
        weight(runtime, "_pose_token_emb.weight", reason), token_input);
    token_embedding = ggml_reshape_2d(context.get(), token_embedding, 256,
                                      positions * batch_size);
    token_embedding = ggml_leaky_relu(context.get(), linear(context.get(), token_embedding,
        weight(runtime, "_proj_pose_token_emb.fc_layers.0.weight", reason),
        weight(runtime, "_proj_pose_token_emb.fc_layers.0.bias", reason)), 0.01F, false);
    token_embedding = ggml_leaky_relu(context.get(), linear(context.get(), token_embedding,
        weight(runtime, "_proj_pose_token_emb.fc_layers.1.weight", reason),
        weight(runtime, "_proj_pose_token_emb.fc_layers.1.bias", reason)), 0.01F, false);
    token_embedding = linear(context.get(), token_embedding,
        weight(runtime, "_proj_pose_token_emb.forward_projection.weight", reason),
        weight(runtime, "_proj_pose_token_emb.forward_projection.bias", reason));
    token_embedding = ggml_reshape_2d(context.get(), token_embedding, 160,
                                      frames * batch_size);

    auto * condition_embedding = linear(context.get(), condition_input,
        weight(runtime, "_proj_local_pose.weight", reason),
        weight(runtime, "_proj_local_pose.bias", reason));
    auto * pose_embedding = ggml_add(context.get(), token_embedding,
        ggml_mul(context.get(), ggml_sub(context.get(), condition_embedding, token_embedding),
                 mask_input));
    pose_embedding = ggml_reshape_2d(context.get(), pose_embedding, 640,
                                     positions * batch_size);
    auto * root_embedding = linear(context.get(),
        ggml_reshape_2d(context.get(), root_input, 16, positions * batch_size),
        weight(runtime, "_proj_local_root_values.weight", reason),
        weight(runtime, "_proj_local_root_values.bias", reason));
    auto * duration_rows = ggml_get_rows(context.get(),
        weight(runtime, "_proj_num_valid_positions.weight", reason), duration_input);
    std::vector<ggml_tensor *> duration_batches;
    duration_batches.reserve(batch_size);
    for (std::uint32_t batch = 0; batch < batch_size; ++batch) {
        auto * row = ggml_view_2d(context.get(), duration_rows, 128, 1,
                                  duration_rows->nb[1], batch * duration_rows->nb[1]);
        auto * shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 128, positions);
        duration_batches.push_back(ggml_repeat(context.get(), row, shape));
    }
    auto * duration_embedding = concat_all(context.get(), duration_batches, 1);
    auto * combined = ggml_concat(context.get(), pose_embedding, root_embedding, 0);
    combined = ggml_concat(context.get(), combined, duration_embedding, 0);
    auto * hidden = ggml_relu(context.get(), linear(context.get(), combined,
        weight(runtime, "_proj_input.0.weight", reason),
        weight(runtime, "_proj_input.0.bias", reason)));
    auto * position_matrix = ggml_reshape_2d(context.get(),
        weight(runtime, "_position_emb.embed", reason), 1024, 16);
    auto * position_view = ggml_view_2d(context.get(), position_matrix, 1024, positions,
                                       position_matrix->nb[1], 0);
    auto * position_shape = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1024,
                                               positions * batch_size);
    hidden = ggml_add(context.get(), hidden,
                      ggml_repeat(context.get(), position_view, position_shape));
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    for (unsigned layer = 0; layer < 16U; ++layer) {
        hidden = transformer_layer_batch(context.get(), runtime, hidden, layer, positions,
                                         batch_size, attention_mask, reason);
        if (hidden == nullptr) return MB_INCOMPATIBLE_MODEL;
    }
    auto * output_tensor = linear(context.get(), hidden,
        weight(runtime, "_proj_pose_output_logit.weight", reason),
        weight(runtime, "_proj_pose_output_logit.bias", reason));
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    auto * graph = ggml_new_graph_custom(context.get(), 4096U + batch_size * 1024U, false);
    ggml_build_forward_expand(graph, output_tensor);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) { reason = "cannot allocate batched pose compute buffer"; return MB_OUT_OF_MEMORY; }

    std::vector<std::int32_t> offset_tokens(pose_tokens.begin(), pose_tokens.end());
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        for (std::uint32_t position = 0; position < positions; ++position)
            for (std::uint32_t head = 0; head < pose_heads; ++head)
                offset_tokens[(static_cast<std::size_t>(batch) * positions + position) * pose_heads + head] +=
                    static_cast<std::int32_t>(head * 11U);
    std::vector<float> mask(has_pose_condition.begin(), has_pose_condition.end());
    std::vector<float> padding_mask(static_cast<std::size_t>(positions) * positions);
    for (std::uint32_t query = 0; query < positions; ++query)
        for (std::uint32_t key = num_tokens; key < positions; ++key)
            padding_mask[static_cast<std::size_t>(query) * positions + key] =
                -std::numeric_limits<float>::infinity();
    std::vector<std::int32_t> duration_indices(
        batch_size, static_cast<std::int32_t>(num_tokens - 6U));
    ggml_backend_tensor_set(token_input, offset_tokens.data(), 0, ggml_nbytes(token_input));
    ggml_backend_tensor_set(root_input, local_root_values.data(), 0, ggml_nbytes(root_input));
    ggml_backend_tensor_set(condition_input, pose_condition.data(), 0, ggml_nbytes(condition_input));
    ggml_backend_tensor_set(mask_input, mask.data(), 0, ggml_nbytes(mask_input));
    ggml_backend_tensor_set(duration_input, duration_indices.data(), 0, ggml_nbytes(duration_input));
    ggml_backend_tensor_set(attention_mask, padding_mask.data(), 0, ggml_nbytes(attention_mask));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("batched pose graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    const auto item_size = static_cast<std::size_t>(positions) * 80U;
    std::vector<float> all_logits(static_cast<std::size_t>(batch_size) * item_size);
    ggml_backend_tensor_get(output_tensor, all_logits.data(), 0,
                            all_logits.size() * sizeof(float));
    logits.assign(batch_size, {});
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        logits[batch].assign(
            all_logits.begin() + static_cast<std::ptrdiff_t>(batch * item_size),
            all_logits.begin() + static_cast<std::ptrdiff_t>((batch + 1U) * item_size));
    return MB_OK;
#endif
}

} // namespace motionbricks::detail
