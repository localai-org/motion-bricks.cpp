#pragma once
#include "neural_runtime.hpp"
#include <array>
#include <map>

namespace motionbricks::detail {
// Raw SOMA model-space interface, matching the seven upstream ONNX inputs.
struct soma_request {
    bool mixed_fp16 = false; // experimental explicit half rounding; parity work in progress
    std::array<float, 40> global_root{};
    std::array<float, 32> local_root{};
    std::array<float, 8*267> poses{};
    std::array<std::uint8_t, 8> has_global{}, has_local{}, has_pose{};
    std::array<std::uint8_t, 11> allowed{};
};
struct soma_result {
    std::uint32_t tokens{};
    bool used_root_predictor{};
    std::vector<float> features; // valid frames only, frame-major [4*tokens,366]
    std::vector<std::int32_t> pose_tokens;
};
using soma_boundaries = std::map<std::string, std::vector<float>>;
mb_status run_soma_features(const neural_runtime &, const soma_request &, soma_result &,
                           std::string &, soma_boundaries * = nullptr);
// Neural batches are grouped by precision and predicted duration. Results and
// optional processing boundaries retain request order; errors publish no output.
mb_status run_soma_features_batch(const neural_runtime &, std::span<const soma_request>,
                                 std::vector<soma_result> &, std::string &,
                                 std::vector<soma_boundaries> * = nullptr);
}
