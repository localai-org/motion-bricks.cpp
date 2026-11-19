#include "neural_runtime.hpp"
#include "pose.hpp"
#include "root.hpp"
#include "decoder.hpp"
#include "soma.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

template<class T> std::vector<T> read(ggml_context * context, const char * name) {
    auto * tensor = ggml_get_tensor(context, name);
    if (!tensor || ggml_element_size(tensor) != sizeof(T)) throw std::runtime_error(name);
    std::vector<T> values(ggml_nelements(tensor));
    std::memcpy(values.data(), tensor->data, values.size() * sizeof(T));
    return values;
}

int main(int argc, char ** argv) try {
    if (argc < 3) throw std::runtime_error("usage: soma-pose-parity BUNDLE FIXTURE [vulkan]");
    const auto device = argc > 3 && std::string(argv[3]) == "vulkan" ? MB_DEVICE_VULKAN : MB_DEVICE_CPU;
    std::shared_ptr<motionbricks::detail::neural_runtime> runtime;
    std::string reason;
    if (motionbricks::detail::create_neural_runtime(argv[1], device, 1, "", runtime, reason) != MB_OK)
        throw std::runtime_error(reason);
    ggml_context * raw = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(argv[2], {false, &raw}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw, ggml_free);
    if (!metadata || !tensors) throw std::runtime_error("invalid fixture");
    auto tokens = read<std::int32_t>(raw, "input.pose_tokens");
    auto roots = read<float>(raw, "input.root_values");
    auto condition = read<float>(raw, "input.pose_cond");
    auto mask32 = read<std::int32_t>(raw, "input.has_pose_cond");
    auto duration = read<std::int32_t>(raw, "input.num_tokens");
    auto expected = read<float>(raw, "output.pose_logits");
    std::vector<std::uint8_t> mask(mask32.begin(), mask32.end());
    const auto positions = static_cast<std::uint32_t>(tokens.size() / 5);
    std::vector<float> actual;
    if (duration.size() != 1 || positions != 16) throw std::runtime_error("expected single-request fixture");
    if (motionbricks::detail::run_pose_planner(*runtime, tokens, roots, condition, mask,
            positions, duration[0], actual, reason) != MB_OK) throw std::runtime_error(reason);
    if (actual.size() != expected.size()) throw std::runtime_error("logit shape mismatch");
    std::vector<motionbricks::detail::neural_tensor_trace> traces;
    std::vector<float> traced_logits;
    {
        motionbricks::detail::neural_trace_scope trace(*runtime, traces, "pose");
        if (motionbricks::detail::run_pose_planner(*runtime, tokens, roots, condition, mask,
                positions, duration[0], traced_logits, reason) != MB_OK) throw std::runtime_error(reason);
    }
    if (traced_logits != actual || traces.empty()) throw std::runtime_error("trace changed pose output or captured nothing");
    double maximum = 0, squared = 0;
    std::size_t mismatches = 0;
    // Only valid token positions are part of the prediction contract.
    const auto valid = static_cast<std::size_t>(duration[0]) * 80;
    for (std::size_t i = 0; i < valid; ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite logit");
        const double error = std::abs(double(actual[i]) - expected[i]);
        maximum = std::max(maximum, error); squared += error * error;
    }
    for (std::size_t i = 0; i < valid; i += 16) {
        const auto a = std::max_element(actual.begin()+i, actual.begin()+i+16) - (actual.begin()+i);
        const auto b = std::max_element(expected.begin()+i, expected.begin()+i+16) - (expected.begin()+i);
        mismatches += a != b;
    }
    // Also verify the independent native batched path against the same oracle.
    std::vector<std::vector<float>> batched;
    if (motionbricks::detail::run_pose_planner_batch(*runtime, tokens, roots, condition, mask,
            positions, duration[0], 1, batched, reason) != MB_OK) throw std::runtime_error(reason);
    double batch_max = 0;
    for (std::size_t i = 0; i < valid; ++i) {
        if (!std::isfinite(batched[0][i])) throw std::runtime_error("non-finite batch logit");
        batch_max = std::max(batch_max, std::abs(double(batched[0][i]) - expected[i]));
    }
    std::cout << "max_abs=" << maximum << " rms=" << std::sqrt(squared/valid)
              << " token_mismatches=" << mismatches << " batch_max_abs=" << batch_max << '\n';
    const auto floats = [&](const char * name) { return read<float>(raw, name); };
    const auto masks = [&](const char * name) {
        const auto values = read<std::int32_t>(raw, name);
        return std::vector<std::uint8_t>(values.begin(), values.end());
    };
    const auto max_error = [](const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() > b.size()) throw std::runtime_error("output shape mismatch");
        double result = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("non-finite output");
            result = std::max(result, std::abs(double(a[i]) - b[i]));
        }
        return result;
    };
    motionbricks::detail::root_result root;
    {
    motionbricks::detail::neural_trace_scope trace(*runtime, traces, "root");
    if (motionbricks::detail::run_root_planner_auto_probe(*runtime,
            floats("root.global"), masks("root.global_mask"), floats("root.local"), masks("root.local_mask"),
            floats("root.poses"), masks("root.pose_mask"), duration[0], root, reason) != MB_OK)
        throw std::runtime_error(reason);
    }
    const auto root_error = max_error(root.global_root_values, floats("root.motion"));
    const auto duration_error = max_error(root.duration_logits, floats("root.logits"));
    std::vector<float> decoded;
    {
    motionbricks::detail::neural_trace_scope trace(*runtime, traces, "decoder");
    if (motionbricks::detail::run_vq_decoder(*runtime, floats("decoder.quantized"),
            floats("decoder.external"), floats("decoder.target"), masks("decoder.mask"), duration[0],
            decoded, reason) != MB_OK) throw std::runtime_error(reason);
    }
    const auto decoder_error = max_error(decoded, floats("decoder.output"));
    motionbricks::detail::soma_request request;
    const auto copy = [&](const char * name, auto & destination) {
        using T = typename std::remove_reference_t<decltype(destination)>::value_type;
        if constexpr (std::is_same_v<T, float>) {
            const auto values = floats(name);
            if (values.size() != destination.size()) throw std::runtime_error("full fixture input size");
            std::copy(values.begin(), values.end(), destination.begin());
        } else {
            const auto values = masks(name);
            if (values.size() != destination.size()) throw std::runtime_error("full fixture mask size");
            std::copy(values.begin(), values.end(), destination.begin());
        }
    };
    copy("full.global_root_values",request.global_root); copy("full.local_root_values",request.local_root);
    copy("full.local_poses",request.poses); copy("full.has_global_root_values",request.has_global);
    copy("full.has_local_root_values",request.has_local); copy("full.has_local_poses",request.has_pose);
    copy("full.allowed_pred_num_tokens",request.allowed);
    motionbricks::detail::soma_result full;
    motionbricks::detail::soma_boundaries boundaries;
    if (motionbricks::detail::run_soma_features(*runtime,request,full,reason,&boundaries)!=MB_OK)
        throw std::runtime_error(reason);
    const auto full_error=max_error(full.features,floats("full.output"));
    if (ggml_get_tensor(raw,"decision.use_root")) {
        const auto expected_decision=floats("decision.use_root");
        if (expected_decision.size()!=1 || (expected_decision[0]!=0 && expected_decision[0]!=1) ||
            full.used_root_predictor!=(expected_decision[0]==1))
            throw std::runtime_error("root-selection decision differs from upstream");
        if (max_error(boundaries.at("root_selection.pose_height_error"),floats("decision.pose_error"))>1e-5 ||
            max_error(boundaries.at("root_selection.root_height_error"),floats("decision.root_error"))>1e-5)
            throw std::runtime_error("root-selection height error differs from upstream");
    }
    std::cout << "full_max_abs=" << full_error << " selected_tokens=" << full.tokens
              << " root_override=" << full.used_root_predictor << '\n';
    std::cout << "root_max_abs=" << root_error << " duration_max_abs=" << duration_error
              << " decoder_max_abs=" << decoder_error << " native_taps=" << traces.size() << '\n';
    if (argc > 4) {
        std::filesystem::create_directories(argv[4]);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({ggml_tensor_overhead() * (traces.size()+1), nullptr, true}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        gguf_set_val_str(file.get(), "trace.format", "motionbricks-neural-trace-v1");
        const auto provenance_key=gguf_find_key(metadata.get(),"motionbricks.source_sha256");
        if(provenance_key<0)throw std::runtime_error("fixture lacks capture provenance");
        gguf_set_val_str(file.get(),"trace.capture_sha256",gguf_get_val_str(metadata.get(),provenance_key));
        for (std::size_t i = 0; i < traces.size(); ++i) {
            auto & trace = traces[i];
            const auto key = "tap." + std::to_string(i);
            auto * tensor = ggml_new_tensor(ctx.get(), static_cast<ggml_type>(trace.type), 4, trace.shape.data());
            tensor->data = trace.data.data();
            ggml_set_name(tensor, key.c_str());
            gguf_add_tensor(file.get(), tensor);
            gguf_set_val_str(file.get(), (key + ".name").c_str(), trace.name.c_str());
            gguf_set_val_str(file.get(), (key + ".operation").c_str(), trace.operation.c_str());
        }
        const auto path = std::filesystem::path(argv[4]) / "native-neural-taps.gguf";
        if (!gguf_write_to_file(file.get(), path.string().c_str(), false)) throw std::runtime_error("cannot write native taps");
        std::unique_ptr<ggml_context, decltype(&ggml_free)> boundary_ctx(
            ggml_init({ggml_tensor_overhead() * (boundaries.size()+1), nullptr, true}), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> boundary_file(gguf_init_empty(), gguf_free);
        gguf_set_val_str(boundary_file.get(), "trace.format", "motionbricks-soma-boundaries-v1");
        gguf_set_val_u32(boundary_file.get(), "soma.tokens", full.tokens);
        for (auto & [name, values] : boundaries) {
            auto * tensor = ggml_new_tensor_1d(boundary_ctx.get(), GGML_TYPE_F32, values.size());
            tensor->data = values.data();
            ggml_set_name(tensor, name.c_str());
            gguf_add_tensor(boundary_file.get(), tensor);
        }
        const auto boundary_path = std::filesystem::path(argv[4]) / "native-boundaries.gguf";
        if (!gguf_write_to_file(boundary_file.get(), boundary_path.string().c_str(), false))
            throw std::runtime_error("cannot write SOMA boundary taps");
    }
    // Initial guard; tighten and broaden on the real-motion fixture suite.
    return maximum < 1e-3 && batch_max < 1e-3 && mismatches == 0 &&
        root_error < 1e-3 && duration_error < 1e-3 && decoder_error < 1e-3 &&
        full_error < 1e-3 && full.tokens==static_cast<unsigned>(duration[0]) ? 0 : 1;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
