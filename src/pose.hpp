#pragma once

#include <motionbricks/motionbricks.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace motionbricks::detail {

class neural_runtime;

mb_status run_pose_planner(const neural_runtime & runtime,
                           std::span<const std::int32_t> pose_tokens,
                           std::span<const float> local_root_values,
                           std::span<const float> pose_condition,
                           std::span<const std::uint8_t> has_pose_condition,
                           std::uint32_t positions,
                           std::uint32_t num_tokens,
                           std::vector<float> & logits,
                           std::string & reason);

mb_status run_pose_planner_batch(
    const neural_runtime & runtime,
    std::span<const std::int32_t> pose_tokens,
    std::span<const float> local_root_values,
    std::span<const float> pose_condition,
    std::span<const std::uint8_t> has_pose_condition,
    std::uint32_t positions, std::uint32_t num_tokens,
    std::uint32_t batch_size, std::vector<std::vector<float>> & logits,
    std::string & reason);

} // namespace motionbricks::detail
