#pragma once

#include <motionbricks/motionbricks.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace motionbricks::detail {

class neural_runtime;

struct decoder_trace {
    std::string name;
    std::vector<float> values;
};

mb_status run_vq_decoder(const neural_runtime & runtime,
                         std::span<const float> quantized,
                         std::span<const float> external_condition,
                         std::span<const float> target_condition,
                         std::span<const std::uint8_t> has_target_condition,
                         std::uint32_t positions,
                         std::vector<float> & motion,
                         std::string & reason,
                         std::vector<decoder_trace> * traces = nullptr);

mb_status run_vq_decoder_batch(
    const neural_runtime & runtime,
    std::span<const float> quantized,
    std::span<const float> external_condition,
    std::span<const float> target_condition,
    std::span<const std::uint8_t> has_target_condition,
    std::uint32_t positions, std::uint32_t batch_size,
    std::vector<std::vector<float>> & motion, std::string & reason);

} // namespace motionbricks::detail
