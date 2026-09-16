#include "decoder.hpp"

#include "neural_runtime.hpp"

#include <array>
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
    auto * result = neural_weight(runtime, "vq-decoder", name);
    if (result == nullptr && reason.empty()) reason = "missing decoder weight: " + name;
    return result;
}

ggml_tensor * linear(ggml_context * context, ggml_tensor * input,
                     ggml_tensor * matrix, ggml_tensor * bias) {
    return ggml_add(context, ggml_mul_mat(context, matrix, input), bias);
}

ggml_tensor * conv(ggml_context * context, ggml_tensor * input,
                   ggml_tensor * kernel, ggml_tensor * bias,
                   int padding, int dilation) {
    // ggml_conv_1d deliberately lowers ordinary F32 convolutions through an
    // F16 im2col buffer.  MotionBricks' residual decoder is sensitive enough
    // for that loss to become visible, so retain F32 through the lowering.
    auto * columns = ggml_im2col(context, kernel, input, 1, 0, padding, 0,
                                dilation, 0, false, GGML_TYPE_F32);
    auto * output = ggml_mul_mat(context,
        ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
        ggml_reshape_2d(context, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    output = ggml_reshape_3d(context, output, columns->ne[1], kernel->ne[2], columns->ne[2]);
    auto * shaped_bias = ggml_reshape_2d(context, bias, 1, bias->ne[0]);
    return ggml_add(context, output, shaped_bias);
}

ggml_tensor * channels_first_to_frames(ggml_context * context, ggml_tensor * input,
                                       std::int64_t channels, std::int64_t frames) {
    return ggml_cont_2d(context, ggml_transpose(context, input), channels, frames);
}

ggml_tensor * frames_to_channels_first(ggml_context * context, ggml_tensor * input,
                                       std::int64_t frames, std::int64_t channels) {
    return ggml_cont_2d(context, ggml_transpose(context, input), frames, channels);
}

ggml_tensor * channels_first_to_frames_batch(ggml_context * context, ggml_tensor * input,
                                             std::int64_t channels, std::int64_t frames,
                                             std::uint32_t batch_size) {
    return ggml_cont_3d(context, ggml_transpose(context, input), channels, frames, batch_size);
}

ggml_tensor * frames_to_channels_first_batch(ggml_context * context, ggml_tensor * input,
                                             std::int64_t frames, std::int64_t channels,
                                             std::uint32_t batch_size) {
    return ggml_cont_3d(context, ggml_transpose(context, input), frames, channels, batch_size);
}

ggml_tensor * concat_all(ggml_context * context,
                         const std::vector<ggml_tensor *> & tensors, int dimension) {
    auto * result = tensors.front();
    for (std::size_t index = 1; index < tensors.size(); ++index)
        result = ggml_concat(context, result, tensors[index], dimension);
    return result;
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

ggml_tensor * residual_stack(ggml_context * context, const neural_runtime & runtime,
                             ggml_tensor * input, unsigned stage, std::string & reason,
                             std::vector<std::pair<std::string, ggml_tensor *>> & traces) {
    static constexpr std::array dilations{27, 9, 3, 1};
    auto * hidden = input;
    for (unsigned block = 0; block < dilations.size(); ++block) {
        const auto prefix = "decoder.model." + std::to_string(stage) + ".0.model." +
                            std::to_string(block) + ".";
        auto * branch = ggml_relu(context, hidden);
        branch = conv(context, branch,
            weight(runtime, prefix + "conv1.weight", reason),
            weight(runtime, prefix + "conv1.bias", reason), dilations[block], dilations[block]);
        traces.emplace_back("model." + std::to_string(stage) + ".0.model." +
                            std::to_string(block) + ".conv1", branch);
        branch = ggml_relu(context, branch);
        branch = conv(context, branch,
            weight(runtime, prefix + "conv2.weight", reason),
            weight(runtime, prefix + "conv2.bias", reason), 0, 1);
        traces.emplace_back("model." + std::to_string(stage) + ".0.model." +
                            std::to_string(block) + ".conv2", branch);
        hidden = ggml_add(context, hidden, branch);
    }
    return hidden;
}

ggml_tensor * residual_stack_batch(ggml_context * context, const neural_runtime & runtime,
                                   ggml_tensor * input, unsigned stage,
                                   std::uint32_t batch_size, std::string & reason) {
    static constexpr std::array dilations{27, 9, 3, 1};
    auto * hidden = input;
    for (unsigned block = 0; block < dilations.size(); ++block) {
        const auto prefix = "decoder.model." + std::to_string(stage) + ".0.model." +
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

#endif

} // namespace

mb_status run_vq_decoder(const neural_runtime & runtime,
                         std::span<const float> quantized,
                         std::span<const float> external_condition,
                         std::span<const float> target_condition,
                         std::span<const std::uint8_t> has_target_condition,
                         std::uint32_t positions,
                         std::vector<float> & motion,
                         std::string & reason,
                         std::vector<decoder_trace> * output_traces) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)quantized; (void)external_condition; (void)target_condition;
    (void)has_target_condition; (void)positions; (void)motion; (void)output_traces;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t latent_width = 256;
    constexpr std::uint32_t external_width = 2;
    constexpr std::uint32_t target_width = 304;
    constexpr std::uint32_t output_width = 413;
    const auto frames = positions * 4U;
    if (positions < 1U || positions > 16U ||
        quantized.size() != static_cast<std::size_t>(positions) * latent_width ||
        external_condition.size() != static_cast<std::size_t>(frames) * external_width ||
        target_condition.size() != static_cast<std::size_t>(frames) * target_width ||
        has_target_condition.size() != frames) {
        reason = "VQ decoder input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    constexpr std::size_t context_bytes = 32U * 1024U * 1024U;
    context_ptr context(ggml_init({context_bytes, nullptr, true}));
    if (!context) {
        reason = "cannot allocate decoder graph metadata";
        return MB_OUT_OF_MEMORY;
    }
    auto * latent_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, positions, latent_width);
    auto * external_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, external_width, frames);
    auto * target_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, target_width, frames);
    auto * target_mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, frames);
    ggml_set_input(latent_input);
    ggml_set_input(external_input);
    ggml_set_input(target_input);
    ggml_set_input(target_mask);

    std::vector<std::pair<std::string, ggml_tensor *>> traces;
    auto * initial_conv = conv(context.get(), latent_input,
        weight(runtime, "decoder.model.0.weight", reason),
        weight(runtime, "decoder.model.0.bias", reason), 1, 1);
    traces.emplace_back("model.0", initial_conv);
    auto * hidden = ggml_relu(context.get(), initial_conv);
    for (unsigned stage_index = 0; stage_index < 2U; ++stage_index) {
        const unsigned stage = stage_index + 2U;
        const auto frame_group = 1U << (2U - stage_index);
        const auto stage_positions = positions * (1U << stage_index);
        const auto stage_frames = stage_positions * frame_group;

        auto * target_embedding = linear(context.get(), target_input,
            weight(runtime, "decoder.target_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "decoder.target_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason));
        traces.emplace_back("target_cond_blocks." + std::to_string(stage_index * 2U), target_embedding);
        auto * hidden_frames = channels_first_to_frames(context.get(), hidden, 512, stage_positions);
        hidden_frames = ggml_reshape_2d(context.get(), hidden_frames, 512 / frame_group, stage_frames);
        hidden_frames = ggml_add(context.get(), hidden_frames,
            ggml_mul(context.get(), ggml_sub(context.get(), ggml_relu(context.get(), target_embedding), hidden_frames),
                     target_mask));
        hidden_frames = ggml_reshape_2d(context.get(), hidden_frames, 512, stage_positions);

        auto * external_grouped = ggml_reshape_2d(
            context.get(), external_input, external_width * frame_group, stage_positions);
        auto * fused = ggml_concat(context.get(), hidden_frames, external_grouped, 0);
        auto * fused_linear = linear(context.get(), fused,
            weight(runtime, "decoder.external_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "decoder.external_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason));
        traces.emplace_back("external_cond_blocks." + std::to_string(stage_index * 2U), fused_linear);
        fused = ggml_relu(context.get(), fused_linear);
        hidden = frames_to_channels_first(context.get(), fused, stage_positions, 512);
        hidden = residual_stack(context.get(), runtime, hidden, stage, reason, traces);
        hidden = ggml_interpolate(context.get(), hidden, hidden->ne[0] * 2, hidden->ne[1],
                                  hidden->ne[2], hidden->ne[3], GGML_SCALE_MODE_NEAREST);
        hidden = conv(context.get(), hidden,
            weight(runtime, "decoder.model." + std::to_string(stage) + ".2.weight", reason),
            weight(runtime, "decoder.model." + std::to_string(stage) + ".2.bias", reason), 1, 1);
        traces.emplace_back("model." + std::to_string(stage) + ".2", hidden);
    }
    auto * post_conv = conv(context.get(), hidden,
        weight(runtime, "decoder.model.4.weight", reason),
        weight(runtime, "decoder.model.4.bias", reason), 1, 1);
    traces.emplace_back("model.4", post_conv);
    hidden = ggml_relu(context.get(), post_conv);
    hidden = conv(context.get(), hidden,
        weight(runtime, "decoder.model.6.weight", reason),
        weight(runtime, "decoder.model.6.bias", reason), 1, 1);
    traces.emplace_back("model.6", hidden);
    auto * output = channels_first_to_frames(context.get(), hidden, output_width, frames);
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    auto * graph = ggml_new_graph_custom(context.get(), 2048, false);
    ggml_build_forward_expand(graph, output);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) {
        reason = "cannot allocate decoder compute buffer";
        return MB_OUT_OF_MEMORY;
    }
    std::vector<float> mask(has_target_condition.begin(), has_target_condition.end());
    ggml_backend_tensor_set(latent_input, quantized.data(), 0, ggml_nbytes(latent_input));
    ggml_backend_tensor_set(external_input, external_condition.data(), 0, ggml_nbytes(external_input));
    ggml_backend_tensor_set(target_input, target_condition.data(), 0, ggml_nbytes(target_input));
    ggml_backend_tensor_set(target_mask, mask.data(), 0, ggml_nbytes(target_mask));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("decoder graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    motion.resize(static_cast<std::size_t>(frames) * output_width);
    ggml_backend_tensor_get(output, motion.data(), 0, motion.size() * sizeof(float));
    if (output_traces != nullptr) {
        output_traces->clear();
        output_traces->reserve(traces.size());
        for (const auto & [name, tensor] : traces) {
            decoder_trace trace{name, std::vector<float>(static_cast<std::size_t>(ggml_nelements(tensor)))};
            ggml_backend_tensor_get(tensor, trace.values.data(), 0, trace.values.size() * sizeof(float));
            output_traces->push_back(std::move(trace));
        }
    }
    return MB_OK;
#endif
}

mb_status run_vq_decoder_batch(
    const neural_runtime & runtime,
    std::span<const float> quantized,
    std::span<const float> external_condition,
    std::span<const float> target_condition,
    std::span<const std::uint8_t> has_target_condition,
    std::uint32_t positions, std::uint32_t batch_size,
    std::vector<std::vector<float>> & motion, std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)runtime; (void)quantized; (void)external_condition; (void)target_condition;
    (void)has_target_condition; (void)positions; (void)batch_size; (void)motion;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    constexpr std::uint32_t latent_width = 256;
    constexpr std::uint32_t external_width = 2;
    constexpr std::uint32_t target_width = 304;
    constexpr std::uint32_t output_width = 413;
    const auto frames = positions * 4U;
    if (batch_size == 0U || batch_size > 64U || positions < 1U || positions > 16U ||
        quantized.size() != static_cast<std::size_t>(batch_size) * positions * latent_width ||
        external_condition.size() != static_cast<std::size_t>(batch_size) * frames * external_width ||
        target_condition.size() != static_cast<std::size_t>(batch_size) * frames * target_width ||
        has_target_condition.size() != static_cast<std::size_t>(batch_size) * frames) {
        reason = "batched VQ decoder input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    context_ptr context(ggml_init({128U * 1024U * 1024U, nullptr, true}));
    if (!context) { reason = "cannot allocate batched decoder graph metadata"; return MB_OUT_OF_MEMORY; }
    auto * latent_input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32,
                                             positions, latent_width, batch_size);
    auto * external_input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32,
                                               external_width, frames, batch_size);
    auto * target_input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32,
                                             target_width, frames, batch_size);
    auto * target_mask = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32,
                                            1, frames, batch_size);
    for (auto * tensor : {latent_input, external_input, target_input, target_mask})
        ggml_set_input(tensor);

    auto * hidden = ggml_relu(context.get(), conv_batch_independent(context.get(), latent_input,
        weight(runtime, "decoder.model.0.weight", reason),
        weight(runtime, "decoder.model.0.bias", reason), 1, 1, batch_size));
    for (unsigned stage_index = 0; stage_index < 2U; ++stage_index) {
        const unsigned stage = stage_index + 2U;
        const auto frame_group = 1U << (2U - stage_index);
        const auto stage_positions = positions * (1U << stage_index);
        const auto stage_frames = stage_positions * frame_group;
        auto * target_embedding = linear(context.get(), target_input,
            weight(runtime, "decoder.target_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "decoder.target_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason));
        auto * hidden_frames = channels_first_to_frames_batch(
            context.get(), hidden, 512, stage_positions, batch_size);
        hidden_frames = ggml_reshape_3d(context.get(), hidden_frames,
                                        512 / frame_group, stage_frames, batch_size);
        hidden_frames = ggml_add(context.get(), hidden_frames,
            ggml_mul(context.get(),
                ggml_sub(context.get(), ggml_relu(context.get(), target_embedding), hidden_frames),
                target_mask));
        hidden_frames = ggml_reshape_3d(context.get(), hidden_frames,
                                        512, stage_positions, batch_size);
        auto * external_grouped = ggml_reshape_3d(
            context.get(), external_input, external_width * frame_group,
            stage_positions, batch_size);
        auto * fused = ggml_concat(context.get(), hidden_frames, external_grouped, 0);
        fused = ggml_relu(context.get(), linear(context.get(), fused,
            weight(runtime, "decoder.external_cond_blocks." + std::to_string(stage_index * 2U) + ".weight", reason),
            weight(runtime, "decoder.external_cond_blocks." + std::to_string(stage_index * 2U) + ".bias", reason)));
        hidden = frames_to_channels_first_batch(
            context.get(), fused, stage_positions, 512, batch_size);
        hidden = residual_stack_batch(context.get(), runtime, hidden, stage, batch_size, reason);
        hidden = ggml_interpolate(context.get(), hidden, hidden->ne[0] * 2, hidden->ne[1],
                                  hidden->ne[2], hidden->ne[3], GGML_SCALE_MODE_NEAREST);
        hidden = conv_batch_independent(context.get(), hidden,
            weight(runtime, "decoder.model." + std::to_string(stage) + ".2.weight", reason),
            weight(runtime, "decoder.model." + std::to_string(stage) + ".2.bias", reason),
            1, 1, batch_size);
    }
    hidden = ggml_relu(context.get(), conv_batch_independent(context.get(), hidden,
        weight(runtime, "decoder.model.4.weight", reason),
        weight(runtime, "decoder.model.4.bias", reason), 1, 1, batch_size));
    hidden = conv_batch_independent(context.get(), hidden,
        weight(runtime, "decoder.model.6.weight", reason),
        weight(runtime, "decoder.model.6.bias", reason), 1, 1, batch_size);
    auto * output_tensor = channels_first_to_frames_batch(
        context.get(), hidden, output_width, frames, batch_size);
    if (!reason.empty()) return MB_INCOMPATIBLE_MODEL;
    auto * graph = ggml_new_graph_custom(context.get(), 4096U + batch_size * 512U, false);
    ggml_build_forward_expand(graph, output_tensor);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), neural_backend(runtime)));
    if (!buffer) { reason = "cannot allocate batched decoder compute buffer"; return MB_OUT_OF_MEMORY; }
    std::vector<float> mask(has_target_condition.begin(), has_target_condition.end());
    ggml_backend_tensor_set(latent_input, quantized.data(), 0, ggml_nbytes(latent_input));
    ggml_backend_tensor_set(external_input, external_condition.data(), 0, ggml_nbytes(external_input));
    ggml_backend_tensor_set(target_input, target_condition.data(), 0, ggml_nbytes(target_input));
    ggml_backend_tensor_set(target_mask, mask.data(), 0, ggml_nbytes(target_mask));
    const auto status = ggml_backend_graph_compute(neural_backend(runtime), graph);
    if (status != GGML_STATUS_SUCCESS) {
        reason = std::string("batched decoder graph failed: ") + ggml_status_to_string(status);
        return MB_COMPUTE_FAILED;
    }
    const auto item_size = static_cast<std::size_t>(frames) * output_width;
    std::vector<float> all_motion(static_cast<std::size_t>(batch_size) * item_size);
    ggml_backend_tensor_get(output_tensor, all_motion.data(), 0, all_motion.size() * sizeof(float));
    motion.assign(batch_size, {});
    for (std::uint32_t batch = 0; batch < batch_size; ++batch)
        motion[batch].assign(
            all_motion.begin() + static_cast<std::ptrdiff_t>(batch * item_size),
            all_motion.begin() + static_cast<std::ptrdiff_t>((batch + 1U) * item_size));
    return MB_OK;
#endif
}

} // namespace motionbricks::detail
