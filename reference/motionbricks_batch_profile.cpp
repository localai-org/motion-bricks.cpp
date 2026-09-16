#include <motionbricks/motionbricks.h>

#include "handles.hpp"
#include "agent.hpp"
#include "decoder.hpp"
#include "planner.hpp"
#include "pose.hpp"
#include "root.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
template <typename T, void (*Free)(T *)>
using handle = std::unique_ptr<T, decltype(Free)>;

std::uint32_t positive(const char * text, const char * name) {
    char * end = nullptr;
    const auto value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value == 0U || value > UINT32_MAX)
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    return static_cast<std::uint32_t>(value);
}
void check(mb_status status, std::string_view operation, const char * error) {
    if (status != MB_OK) throw std::runtime_error(std::string(operation) + ": " + error);
}
double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}
double maximum_difference(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) return INFINITY;
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index)
        result = std::max(result, std::abs(static_cast<double>(left[index]) - right[index]));
    return result;
}
} // namespace

int main(int argc, char ** argv) try {
    if (argc != 7 && argc != 8) {
        std::cerr << "usage: motionbricks-batch-profile BUNDLE STYLE THREADS BATCH WARMUP ROUNDS [cpu|vulkan]\n";
        return 2;
    }
    const auto threads = positive(argv[3], "threads");
    const auto batch_size = positive(argv[4], "batch");
    const auto warmup = positive(argv[5], "warmup");
    const auto rounds = positive(argv[6], "rounds");
    const std::string_view device_name = argc == 8 ? argv[7] : "cpu";
    const auto device = device_name == "cpu" ? MB_DEVICE_CPU :
                        device_name == "vulkan" ? MB_DEVICE_VULKAN : UINT32_MAX;
    if (device == UINT32_MAX) throw std::runtime_error("device must be cpu or vulkan");
    if (batch_size > 64U) throw std::runtime_error("batch must not exceed 64");
    std::array<char, 1024> error{};
    mb_runtime_options * raw_options = nullptr;
    check(mb_runtime_options_create(&raw_options, error.data(), error.size()), "options", error.data());
    handle<mb_runtime_options, mb_runtime_options_free> options(raw_options, mb_runtime_options_free);
    check(mb_runtime_options_set_device(options.get(), device, error.data(), error.size()),
          "device", error.data());
    check(mb_runtime_options_set_threads(options.get(), threads, error.data(), error.size()),
          "threads", error.data());
    mb_model * raw_model = nullptr;
    check(mb_model_load(argv[1], options.get(), &raw_model, error.data(), error.size()),
          "model", error.data());
    handle<mb_model, mb_model_free> model(raw_model, mb_model_free);
    mb_style * raw_style = nullptr;
    check(mb_style_load(model.get(), argv[2], &raw_style, error.data(), error.size()),
          "style", error.data());
    handle<mb_style, mb_style_free> style(raw_style, mb_style_free);

    std::vector<handle<mb_agent, mb_agent_free>> agents;
    std::vector<handle<mb_command, mb_command_free>> commands;
    std::vector<mb_agent *> agent_ptrs;
    std::vector<const mb_command *> command_ptrs;
    for (std::uint32_t index = 0; index < batch_size; ++index) {
        mb_agent * agent = nullptr; mb_command * command = nullptr;
        check(mb_agent_create(model.get(), &agent, error.data(), error.size()), "agent", error.data());
        agents.emplace_back(agent, mb_agent_free);
        check(mb_agent_reset(agent, style.get(), error.data(), error.size()), "reset", error.data());
        check(mb_command_create(&command, error.data(), error.size()), "command", error.data());
        commands.emplace_back(command, mb_command_free);
        check(mb_command_set_style(command, style.get(), error.data(), error.size()), "command style", error.data());
        check(mb_command_set_movement_direction(command, 0.0F, 0.0F, 1.0F,
                                                error.data(), error.size()), "movement", error.data());
        check(mb_command_set_facing_direction(command, 0.0F, 0.0F, 1.0F,
                                              error.data(), error.size()), "facing", error.data());
        check(mb_command_set_seed(command, UINT64_C(1009) + index,
                                  error.data(), error.size()), "seed", error.data());
        agent_ptrs.push_back(agent); command_ptrs.push_back(command);
    }
    const auto serial_round = [&]() {
        std::vector<handle<mb_motion, mb_motion_free>> motions;
        const auto start = clock_type::now();
        for (std::uint32_t index = 0; index < batch_size; ++index) {
            mb_motion * motion = nullptr;
            check(mb_agent_plan(agent_ptrs[index], command_ptrs[index], &motion,
                                error.data(), error.size()), "serial plan", error.data());
            motions.emplace_back(motion, mb_motion_free);
        }
        const auto stop = clock_type::now();
        return std::chrono::duration<double, std::milli>(stop - start).count();
    };
    const auto batch_round = [&](bool native,
                                 std::vector<handle<mb_motion, mb_motion_free>> * retained) {
        std::vector<mb_motion *> raw(batch_size);
        std::vector<handle<mb_motion, mb_motion_free>> motions;
        std::string reason;
        if (native) {
            for (std::uint32_t index = 0; index < batch_size; ++index) {
                auto * motion = new mb_motion;
                raw[index] = motion;
                motions.emplace_back(motion, mb_motion_free);
            }
        }
        const auto start = clock_type::now();
        if (native) {
            check(motionbricks::detail::plan_agent_batch(
                agent_ptrs, command_ptrs, raw, reason, true), "native batch plan", reason.c_str());
        } else {
            check(mb_agent_plan_batch(agent_ptrs.data(), command_ptrs.data(), batch_size, raw.data(),
                                      error.data(), error.size()), "batch plan", error.data());
        }
        const auto stop = clock_type::now();
        if (!native) for (auto * motion : raw) motions.emplace_back(motion, mb_motion_free);
        if (retained != nullptr) *retained = std::move(motions);
        return std::chrono::duration<double, std::milli>(stop - start).count();
    };
    for (std::uint32_t index = 0; index < warmup; ++index) {
        (void)serial_round(); (void)batch_round(false, nullptr); (void)batch_round(true, nullptr);
    }
    std::vector<double> serial_ms, batch_ms, native_batch_ms;
    for (std::uint32_t index = 0; index < rounds; ++index) serial_ms.push_back(serial_round());
    for (std::uint32_t index = 0; index < rounds; ++index) batch_ms.push_back(batch_round(false, nullptr));
    for (std::uint32_t index = 0; index < rounds; ++index) native_batch_ms.push_back(batch_round(true, nullptr));

    std::vector<handle<mb_motion, mb_motion_free>> serial_outputs;
    for (std::uint32_t index = 0; index < batch_size; ++index) {
        mb_motion * motion = nullptr;
        check(mb_agent_plan(agent_ptrs[index], command_ptrs[index], &motion,
                            error.data(), error.size()), "parity serial plan", error.data());
        serial_outputs.emplace_back(motion, mb_motion_free);
    }
    std::vector<handle<mb_motion, mb_motion_free>> batch_outputs;
    (void)batch_round(true, &batch_outputs);
    double max_root = 0.0, max_rotation = 0.0;
    bool frames_match = true;
    for (std::uint32_t index = 0; index < batch_size; ++index) {
        frames_match = frames_match && serial_outputs[index]->frames == batch_outputs[index]->frames;
        max_root = std::max(max_root, maximum_difference(
            serial_outputs[index]->root_translations, batch_outputs[index]->root_translations));
        max_rotation = std::max(max_rotation, maximum_difference(
            serial_outputs[index]->local_rotations_xyzw, batch_outputs[index]->local_rotations_xyzw));
    }
    std::vector<motionbricks::detail::transition_trace> traces(batch_size);
    for (std::uint32_t index = 0; index < batch_size; ++index) {
        mb_motion ignored; std::string reason;
        check(motionbricks::detail::plan_agent_trace(
            *agent_ptrs[index], *command_ptrs[index], ignored, traces[index], reason),
            "trace plan", reason.c_str());
    }
    const auto tokens = traces.front().selected_tokens;
    bool same_tokens = std::all_of(traces.begin(), traces.end(),
        [&](const auto & trace) { return trace.selected_tokens == tokens; });
    const auto append = []<typename T>(std::vector<T> & destination, const auto & source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    std::vector<float> normalized_global, normalized_local, normalized_poses;
    std::vector<std::uint8_t> global_mask, local_mask, pose_mask;
    for (const auto & trace : traces) {
        append(normalized_global, trace.normalized_global_root);
        append(normalized_local, trace.normalized_local_root);
        append(normalized_poses, trace.normalized_poses);
        append(global_mask, trace.constraints.has_global_root);
        append(local_mask, trace.constraints.has_local_root);
        append(pose_mask, trace.constraints.has_poses);
    }
    std::string diagnostic_reason;
    std::vector<motionbricks::detail::root_result> root_batch;
    check(motionbricks::detail::run_root_planner_auto_probe_batch(
        *model->runtime, normalized_global, global_mask, normalized_local, local_mask,
        normalized_poses, pose_mask, tokens, batch_size, root_batch, diagnostic_reason),
        "root batch diagnostic", diagnostic_reason.c_str());
    double root_stage_diff = 0.0;
    std::vector<double> root_stage_diffs(batch_size);
    for (std::uint32_t index = 0; index < batch_size; ++index)
        root_stage_diff = std::max(root_stage_diff, root_stage_diffs[index] = maximum_difference(
            traces[index].predicted_global_root, root_batch[index].global_root_values));

    std::vector<std::int32_t> initial_pose_tokens(
        static_cast<std::size_t>(batch_size) * tokens * 8U, 10);
    std::vector<float> pose_roots, pose_conditions;
    std::vector<std::uint8_t> pose_condition_masks;
    for (const auto & trace : traces) {
        append(pose_roots, trace.pose_root_condition);
        append(pose_conditions, trace.pose_condition);
        append(pose_condition_masks, trace.has_pose_condition);
    }
    std::vector<std::vector<float>> pose_logits;
    check(motionbricks::detail::run_pose_planner_batch(
        *model->runtime, initial_pose_tokens, pose_roots, pose_conditions,
        pose_condition_masks, tokens, tokens, batch_size, pose_logits, diagnostic_reason),
        "pose batch diagnostic", diagnostic_reason.c_str());
    double pose_stage_diff = 0.0;
    std::vector<double> pose_stage_diffs(batch_size);
    for (std::uint32_t index = 0; index < batch_size; ++index)
        pose_stage_diff = std::max(pose_stage_diff, pose_stage_diffs[index] = maximum_difference(
            traces[index].pose_logits, pose_logits[index]));

    std::vector<float> decoder_quantized, decoder_external, decoder_condition;
    std::vector<std::uint8_t> decoder_masks;
    for (const auto & trace : traces) {
        append(decoder_quantized, trace.decoder_quantized);
        append(decoder_external, trace.decoder_external_condition);
        append(decoder_condition, trace.pose_condition);
        append(decoder_masks, trace.decoder_target_mask);
    }
    std::vector<std::vector<float>> decoder_outputs;
    check(motionbricks::detail::run_vq_decoder_batch(
        *model->runtime, decoder_quantized, decoder_external, decoder_condition,
        decoder_masks, tokens, batch_size, decoder_outputs, diagnostic_reason),
        "decoder batch diagnostic", diagnostic_reason.c_str());
    double decoder_stage_diff = 0.0;
    std::vector<double> decoder_stage_diffs(batch_size);
    for (std::uint32_t index = 0; index < batch_size; ++index)
        decoder_stage_diff = std::max(decoder_stage_diff, decoder_stage_diffs[index] = maximum_difference(
            traces[index].decoder_output, decoder_outputs[index]));
    const auto serial_mean = mean(serial_ms), batch_mean = mean(batch_ms);
    const auto native_batch_mean = mean(native_batch_ms);
    std::cout << std::fixed << std::setprecision(9)
              << "{\"device\":\"" << device_name << "\",\"threads\":" << threads
              << ",\"batch_size\":" << batch_size
              << ",\"serial_round_mean_ms\":" << serial_mean
              << ",\"batch_round_mean_ms\":" << batch_mean
              << ",\"speedup\":" << serial_mean / batch_mean
              << ",\"batch_plans_per_second\":" << 1000.0 * batch_size / batch_mean
              << ",\"native_batch_round_mean_ms\":" << native_batch_mean
              << ",\"native_batch_speedup\":" << serial_mean / native_batch_mean
              << ",\"frames_match\":" << (frames_match ? "true" : "false")
              << ",\"max_root_abs_diff\":" << max_root
              << ",\"max_rotation_abs_diff\":" << max_rotation
              << ",\"same_tokens\":" << (same_tokens ? "true" : "false")
              << ",\"root_stage_max_abs_diff\":" << root_stage_diff
              << ",\"pose_stage_max_abs_diff\":" << pose_stage_diff
              << ",\"decoder_stage_max_abs_diff\":" << decoder_stage_diff
              << ",\"root_lane_diffs\":[";
    for (std::size_t index = 0; index < root_stage_diffs.size(); ++index) {
        if (index != 0U) std::cout << ','; std::cout << root_stage_diffs[index];
    }
    std::cout << "],\"pose_lane_diffs\":[";
    for (std::size_t index = 0; index < pose_stage_diffs.size(); ++index) {
        if (index != 0U) std::cout << ','; std::cout << pose_stage_diffs[index];
    }
    std::cout << "],\"decoder_lane_diffs\":[";
    for (std::size_t index = 0; index < decoder_stage_diffs.size(); ++index) {
        if (index != 0U) std::cout << ','; std::cout << decoder_stage_diffs[index];
    }
    std::cout << "]}\n";
    return frames_match && same_tokens && max_root < 1.0e-4 && max_rotation < 1.0e-4 &&
        root_stage_diff < 1.0e-4 && pose_stage_diff < 1.0e-4 && decoder_stage_diff < 1.0e-4 ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << "motionbricks-batch-profile: " << exception.what() << '\n';
    return 1;
}
