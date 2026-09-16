#include "model.hpp"

#include "handles.hpp"
#include "neural_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml.h>
#include <gguf.h>
#endif

namespace motionbricks::detail {
namespace {

#if defined(MOTIONBRICKS_HAVE_GGML)

constexpr std::string_view upstream_revision = "a0732b642c0333077e127a2f56ab0014c196bca4";
constexpr std::uint64_t inference_parameters = UINT64_C(183148382);

struct component_spec {
    const char * name;
    const char * filename;
    const char * source_hash;
    std::int64_t tensors;
    std::uint64_t parameters;
    bool load_data;
};

constexpr std::array components{
    component_spec{"pose", "pose.gguf",
        "01327768be7413111dc927947a95cfb3e9ee7c52acfa35a054e8c2f3b838b888",
        209, UINT64_C(136588272), false},
    component_spec{"root", "root.gguf",
        "d1529a8c9da915cb7dd0499272baba61db480660c0aae92b67fbe69828e83c5a",
        150, UINT64_C(34122833), false},
    component_spec{"vq-decoder", "vq-decoder.gguf",
        "544782e605ed96d60bf999243ef8f44640ea021a5655ef20af2d2853c3e18b5b",
        51, UINT64_C(12437277), false},
    component_spec{"support", "support.gguf",
        "229b764411652b2ab0f824481d6daf897f701a97223029759444fc5bc241ea22",
        4, UINT64_C(972), true},
};

struct ggml_deleter {
    void operator()(ggml_context * value) const noexcept { ggml_free(value); }
};

struct gguf_deleter {
    void operator()(gguf_context * value) const noexcept { gguf_free(value); }
};

using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

bool metadata_string(const gguf_context * context, const char * key,
                     std::string_view expected, std::string & reason) {
    const auto index = gguf_find_key(context, key);
    if (index < 0 || gguf_get_kv_type(context, index) != GGUF_TYPE_STRING) {
        reason = std::string("missing string metadata: ") + key;
        return false;
    }
    if (std::string_view(gguf_get_val_str(context, index)) != expected) {
        reason = std::string("incompatible metadata value: ") + key;
        return false;
    }
    return true;
}

bool metadata_u32(const gguf_context * context, const char * key,
                  std::uint32_t expected, std::string & reason) {
    const auto index = gguf_find_key(context, key);
    if (index < 0 || gguf_get_kv_type(context, index) != GGUF_TYPE_UINT32) {
        reason = std::string("missing uint32 metadata: ") + key;
        return false;
    }
    if (gguf_get_val_u32(context, index) != expected) {
        reason = std::string("incompatible metadata value: ") + key;
        return false;
    }
    return true;
}

bool metadata_u64(const gguf_context * context, const char * key,
                  std::uint64_t expected, std::string & reason) {
    const auto index = gguf_find_key(context, key);
    if (index < 0 || gguf_get_kv_type(context, index) != GGUF_TYPE_UINT64) {
        reason = std::string("missing uint64 metadata: ") + key;
        return false;
    }
    if (gguf_get_val_u64(context, index) != expected) {
        reason = std::string("incompatible metadata value: ") + key;
        return false;
    }
    return true;
}

bool tensor_shape(const gguf_context * context, const char * name,
                  std::array<std::int64_t, 4> expected, ggml_type type,
                  std::string & reason) {
    const auto index = gguf_find_tensor(context, name);
    if (index < 0) {
        reason = std::string("missing tensor: ") + name;
        return false;
    }
    if (gguf_get_tensor_type(context, index) != type) {
        reason = std::string("incorrect tensor type: ") + name;
        return false;
    }
    const auto * dimensions = gguf_get_tensor_ne(context, index);
    if (!std::equal(expected.begin(), expected.end(), dimensions)) {
        reason = std::string("incorrect tensor shape: ") + name;
        return false;
    }
    return true;
}

std::vector<std::string> split_names(std::string_view value) {
    std::vector<std::string> result;
    while (!value.empty()) {
        const auto comma = value.find(',');
        result.emplace_back(value.substr(0, comma));
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
    }
    return result;
}

mb_status load_component(const std::filesystem::path & path, const component_spec & spec,
                         mb_model & output, std::string & reason) {
    if (!std::filesystem::is_regular_file(path)) {
        reason = "missing GGUF component: " + path.string();
        return MB_IO_ERROR;
    }

    ggml_context * raw_ggml = nullptr;
    const gguf_init_params params{!spec.load_data, &raw_ggml};
    gguf_ptr gguf(gguf_init_from_file(path.string().c_str(), params));
    ggml_ptr ggml(raw_ggml);
    if (!gguf || !ggml) {
        reason = "could not parse GGUF component: " + path.string();
        return MB_INVALID_FORMAT;
    }
    if (gguf_get_version(gguf.get()) != 3U) {
        reason = "unsupported GGUF version in " + path.string();
        return MB_INVALID_FORMAT;
    }
    if (!metadata_string(gguf.get(), "general.architecture", "motionbricks", reason) ||
        !metadata_u32(gguf.get(), "motionbricks.format_version", 1U, reason) ||
        !metadata_string(gguf.get(), "motionbricks.component", spec.name, reason) ||
        !metadata_string(gguf.get(), "motionbricks.skeleton", "g1skel34", reason) ||
        !metadata_string(gguf.get(), "motionbricks.upstream_revision", upstream_revision, reason) ||
        !metadata_string(gguf.get(), "motionbricks.source_sha256", spec.source_hash, reason) ||
        !metadata_u64(gguf.get(), "motionbricks.parameter_count", spec.parameters, reason)) {
        reason += " (" + path.string() + ")";
        return MB_INCOMPATIBLE_MODEL;
    }
    if (gguf_get_n_tensors(gguf.get()) != spec.tensors) {
        reason = "incorrect tensor count in " + path.string();
        return MB_INCOMPATIBLE_MODEL;
    }

    std::unordered_set<std::string_view> names;
    for (std::int64_t index = 0; index < spec.tensors; ++index) {
        const std::string_view name = gguf_get_tensor_name(gguf.get(), index);
        if (name.empty() || !names.insert(name).second) {
            reason = "invalid or duplicate tensor name in " + path.string();
            return MB_INVALID_FORMAT;
        }
    }

    if (std::string_view(spec.name) == "pose") {
        if (!tensor_shape(gguf.get(), "_pose_token_emb.weight", {32, 88, 1, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "_position_emb.embed", {1024, 1, 16, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "_proj_pose_output_logit.weight", {1024, 80, 1, 1}, GGML_TYPE_F32, reason))
            return MB_INCOMPATIBLE_MODEL;
    } else if (std::string_view(spec.name) == "root") {
        if (!tensor_shape(gguf.get(), "_position_emb.embed", {512, 1, 16, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "_proj_num_token_output_logit.weight", {512, 12, 1, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "_conv_output.model.6.weight", {3, 512, 5, 1}, GGML_TYPE_F32, reason))
            return MB_INCOMPATIBLE_MODEL;
    } else if (std::string_view(spec.name) == "vq-decoder") {
        if (!tensor_shape(gguf.get(), "quantizer.vq._codebook.embed", {32, 10, 8, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "decoder.model.6.weight", {3, 512, 413, 1}, GGML_TYPE_F32, reason))
            return MB_INCOMPATIBLE_MODEL;
    } else {
        if (!tensor_shape(gguf.get(), "neutral_joints", {3, 34, 1, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "joint_parents", {34, 1, 1, 1}, GGML_TYPE_I32, reason) ||
            !tensor_shape(gguf.get(), "motion_mean", {418, 1, 1, 1}, GGML_TYPE_F32, reason) ||
            !tensor_shape(gguf.get(), "motion_std", {418, 1, 1, 1}, GGML_TYPE_F32, reason))
            return MB_INCOMPATIBLE_MODEL;

        const auto joint_names_key = gguf_find_key(gguf.get(), "motionbricks.joint_names");
        if (joint_names_key < 0 || gguf_get_kv_type(gguf.get(), joint_names_key) != GGUF_TYPE_STRING) {
            reason = "missing joint names in support component";
            return MB_INCOMPATIBLE_MODEL;
        }
        output.joint_names = split_names(gguf_get_val_str(gguf.get(), joint_names_key));
        if (output.joint_names.size() != 34U ||
            std::unordered_set<std::string>(output.joint_names.begin(), output.joint_names.end()).size() != 34U) {
            reason = "invalid joint names in support component";
            return MB_INCOMPATIBLE_MODEL;
        }
        const auto * parents_tensor = ggml_get_tensor(ggml.get(), "joint_parents");
        const auto * parents = static_cast<const std::int32_t *>(parents_tensor->data);
        output.joint_parents.assign(parents, parents + 34);
        const auto copy_f32 = [&](const char * name, std::size_t count,
                                  std::vector<float> & destination) {
            const auto * tensor = ggml_get_tensor(ggml.get(), name);
            const auto * values = static_cast<const float *>(tensor->data);
            destination.assign(values, values + count);
        };
        copy_f32("neutral_joints", 34U * 3U, output.neutral_joints);
        copy_f32("motion_mean", 418U, output.motion_mean);
        copy_f32("motion_std", 418U, output.motion_std);
        for (std::size_t index = 0; index < output.joint_parents.size(); ++index) {
            const auto parent = output.joint_parents[index];
            if ((index == 0U && parent != -1) ||
                (index != 0U && (parent < 0 || static_cast<std::size_t>(parent) >= index))) {
                reason = "invalid parent topology in support component";
                return MB_INCOMPATIBLE_MODEL;
            }
        }
        if (std::any_of(output.neutral_joints.begin(), output.neutral_joints.end(),
                        [](float value) { return !std::isfinite(value); }) ||
            std::any_of(output.motion_mean.begin(), output.motion_mean.end(),
                        [](float value) { return !std::isfinite(value); }) ||
            std::any_of(output.motion_std.begin(), output.motion_std.end(),
                        [](float value) { return !std::isfinite(value) || value < 0.0F; })) {
            reason = "invalid support values";
            return MB_INCOMPATIBLE_MODEL;
        }
    }
    return MB_OK;
}

#endif

} // namespace

mb_status load_model_bundle(const std::filesystem::path & directory,
                            const mb_runtime_options * options,
                            mb_model & output, std::string & reason) {
#if !defined(MOTIONBRICKS_HAVE_GGML)
    (void)directory;
    (void)output;
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
#else
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
        reason = error ? "cannot access model bundle: " + error.message()
                       : "model bundle is not a directory: " + directory.string();
        return MB_IO_ERROR;
    }
    mb_model temporary;
    temporary.bundle_directory = std::filesystem::absolute(directory, error).lexically_normal().string();
    if (error) {
        reason = "cannot resolve model bundle: " + error.message();
        return MB_IO_ERROR;
    }
    for (const auto & component : components) {
        const auto status = load_component(directory / component.filename, component, temporary, reason);
        if (status != MB_OK) return status;
    }
    temporary.parameter_count = inference_parameters;
    const auto status = create_neural_runtime(
        directory, options != nullptr ? options->device : MB_DEVICE_CPU,
        options != nullptr ? options->threads : 0U,
        options != nullptr ? options->backend_directory : std::string{},
        temporary.runtime, reason);
    if (status != MB_OK) return status;
    if (!neural_copy_f32(*temporary.runtime, "vq-decoder", "quantizer.vq._codebook.embed",
                         temporary.vq_codebook, reason) ||
        temporary.vq_codebook.size() != 8U * 10U * 32U) {
        if (reason.empty()) reason = "VQ codebook shape mismatch";
        return MB_INCOMPATIBLE_MODEL;
    }
    output = std::move(temporary);
    return MB_OK;
#endif
}

} // namespace motionbricks::detail
