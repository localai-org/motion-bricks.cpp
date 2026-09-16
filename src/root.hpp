#pragma once

#include <motionbricks/motionbricks.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace motionbricks::detail {

class neural_runtime;

struct root_result {
    std::uint32_t tokens = 0;
    std::vector<float> duration_logits;
    std::vector<float> global_root_values;
};

mb_status run_root_duration_probe(
    const neural_runtime & runtime,
    std::span<const float> global_root_values,
    std::span<const std::uint8_t> has_global_root_values,
    std::span<const float> local_root_values,
    std::span<const std::uint8_t> has_local_root_values,
    std::span<const float> poses,
    std::span<const std::uint8_t> has_poses,
    std::vector<float> & duration_logits, std::string & reason);

mb_status run_root_duration_probe_batch(
    const neural_runtime & runtime,
    std::span<const float> global_root_values,
    std::span<const std::uint8_t> has_global_root_values,
    std::span<const float> local_root_values,
    std::span<const std::uint8_t> has_local_root_values,
    std::span<const float> poses,
    std::span<const std::uint8_t> has_poses,
    std::uint32_t batch_size, std::vector<std::vector<float>> & duration_logits,
    std::string & reason);

mb_status run_root_planner(const neural_runtime & runtime,
                           std::span<const float> global_root_values,
                           std::span<const std::uint8_t> has_global_root_values,
                           std::span<const float> local_root_values,
                           std::span<const std::uint8_t> has_local_root_values,
                           std::span<const float> poses,
                           std::span<const std::uint8_t> has_poses,
                           std::uint32_t requested_tokens,
                           root_result & output,
                           std::string & reason);

/* Uses the masked-duration embedding (upstream token value 18) while building
   a graph for candidate_tokens. This is the first pass of auto-duration. */
mb_status run_root_planner_auto_probe(const neural_runtime & runtime,
                                      std::span<const float> global_root_values,
                                      std::span<const std::uint8_t> has_global_root_values,
                                      std::span<const float> local_root_values,
                                      std::span<const std::uint8_t> has_local_root_values,
                                      std::span<const float> poses,
                                      std::span<const std::uint8_t> has_poses,
                                      std::uint32_t candidate_tokens,
                                      root_result & output,
                                      std::string & reason);

/* Batch-major inputs. Each batch item is contiguous and has the same shape as
   the corresponding batch-one argument. All items use candidate_tokens. */
mb_status run_root_planner_auto_probe_batch(
    const neural_runtime & runtime,
    std::span<const float> global_root_values,
    std::span<const std::uint8_t> has_global_root_values,
    std::span<const float> local_root_values,
    std::span<const std::uint8_t> has_local_root_values,
    std::span<const float> poses,
    std::span<const std::uint8_t> has_poses,
    std::uint32_t candidate_tokens, std::uint32_t batch_size,
    std::vector<root_result> & output, std::string & reason);

} // namespace motionbricks::detail
