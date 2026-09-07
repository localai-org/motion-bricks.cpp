#include "neural_runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml-cpu.h>
#include <gguf.h>
#if defined(MOTIONBRICKS_HAVE_VULKAN)
#include <ggml-vulkan.h>
#endif
#endif

namespace motionbricks::detail {

#if defined(MOTIONBRICKS_HAVE_GGML)
namespace {

struct ggml_context_deleter {
    void operator()(ggml_context * value) const noexcept { ggml_free(value); }
};
struct gguf_context_deleter {
    void operator()(gguf_context * value) const noexcept { gguf_free(value); }
};
struct backend_deleter {
    void operator()(ggml_backend * value) const noexcept { ggml_backend_free(value); }
};
struct buffer_deleter {
    void operator()(ggml_backend_buffer * value) const noexcept { ggml_backend_buffer_free(value); }
};

using context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;
using gguf_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;
using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

bool copy_gguf_data(const std::filesystem::path & path, ggml_context * tensors,
                    const gguf_context * metadata, std::string & reason) {
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
        std::fopen(path.string().c_str(), "rb"), &std::fclose);
    if (!file) {
        reason = "cannot open GGUF weights: " + path.string();
        return false;
    }
    std::vector<std::byte> staging(4U * 1024U * 1024U);
    const auto count = gguf_get_n_tensors(metadata);
    for (std::int64_t index = 0; index < count; ++index) {
        const char * name = gguf_get_tensor_name(metadata, index);
        auto * tensor = ggml_get_tensor(tensors, name);
        if (tensor == nullptr) {
            reason = std::string("missing allocated tensor: ") + name;
            return false;
        }
        const auto offset = gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, index);
        if (offset > static_cast<std::size_t>(std::numeric_limits<long>::max()) ||
            std::fseek(file.get(), static_cast<long>(offset), SEEK_SET) != 0) {
            reason = std::string("cannot seek to tensor data: ") + name;
            return false;
        }
        const auto bytes = ggml_nbytes(tensor);
        for (std::size_t position = 0; position < bytes; position += staging.size()) {
            const auto amount = std::min(staging.size(), bytes - position);
            if (std::fread(staging.data(), 1, amount, file.get()) != amount) {
                reason = std::string("truncated tensor data: ") + name;
                return false;
            }
            ggml_backend_tensor_set(tensor, staging.data(), position, amount);
        }
    }
    return true;
}

} // namespace

class neural_runtime {
public:
    struct component {
        std::string name;
        context_ptr context;
        buffer_ptr buffer;
    };

    backend_ptr backend;
    std::vector<component> components;
};

namespace {

mb_status open_component(neural_runtime & runtime, const std::filesystem::path & path,
                         std::string name, std::string & reason) {
    ggml_context * raw_context = nullptr;
    const gguf_init_params params{true, &raw_context};
    gguf_ptr metadata(gguf_init_from_file(path.string().c_str(), params));
    context_ptr context(raw_context);
    if (!metadata || !context) {
        reason = "cannot initialize GGUF weights: " + path.string();
        return MB_INVALID_FORMAT;
    }
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), runtime.backend.get()));
    if (!buffer) {
        reason = "cannot allocate backend weights for " + path.string();
        return MB_OUT_OF_MEMORY;
    }
    if (!copy_gguf_data(path, context.get(), metadata.get(), reason)) return MB_IO_ERROR;
    runtime.components.push_back({std::move(name), std::move(context), std::move(buffer)});
    return MB_OK;
}

} // namespace

static mb_status create_runtime(const std::filesystem::path & bundle,
                                mb_device device, std::uint32_t threads,
                                const std::string & backend_directory,
                                std::shared_ptr<neural_runtime> & output,
                                std::string & reason, bool sonic) {
    output.reset();
    if (!backend_directory.empty()) ggml_backend_load_all_from_path(backend_directory.c_str());
    auto runtime = std::make_shared<neural_runtime>();
    const auto selected = device == MB_DEVICE_AUTO ? MB_DEVICE_CPU : device;
    if (selected == MB_DEVICE_CPU) {
        runtime->backend.reset(ggml_backend_cpu_init());
        if (runtime->backend) {
            const auto hardware = std::max(1U, std::thread::hardware_concurrency());
            const auto count = threads == 0U ? hardware : threads;
            ggml_backend_cpu_set_n_threads(runtime->backend.get(), static_cast<int>(count));
        }
    } else if (selected == MB_DEVICE_VULKAN) {
#if defined(MOTIONBRICKS_HAVE_VULKAN)
        // GGML's Vulkan F16/cooperative-matrix fast paths may be selected even
        // for F32 weights.  The released MotionBricks baseline is explicitly
        // F32, and those paths create material decoder drift after many
        // residual convolutions.  Keep the parity runtime strictly F32.
#if defined(_WIN32)
        _putenv_s("GGML_VK_DISABLE_F16", "1");
        _putenv_s("GGML_VK_DISABLE_COOPMAT", "1");
        _putenv_s("GGML_VK_DISABLE_COOPMAT2", "1");
#else
        setenv("GGML_VK_DISABLE_F16", "1", 0);
        setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
        setenv("GGML_VK_DISABLE_COOPMAT2", "1", 0);
#endif
        try {
            // ggml_backend_vk_init(0) asserts if discovery returned no devices.
            // Driver/sandbox failures must be an API error, not an ABI abort.
            if (ggml_backend_vk_get_device_count() == 0) {
                reason = "no accessible Vulkan devices";
                return MB_BACKEND_UNAVAILABLE;
            }
            runtime->backend.reset(ggml_backend_vk_init(0));
        } catch (const std::exception & e) {
            reason = std::string("cannot initialize Vulkan backend: ") + e.what();
            return MB_BACKEND_UNAVAILABLE;
        }
#else
        reason = "this build has no Vulkan backend";
        return MB_BACKEND_UNAVAILABLE;
#endif
    }
    if (!runtime->backend) {
        reason = selected == MB_DEVICE_VULKAN ? "cannot initialize Vulkan backend"
                                               : "cannot initialize CPU backend";
        return MB_BACKEND_UNAVAILABLE;
    }
    if (sonic) {
        const auto status = open_component(*runtime, bundle, "sonic", reason);
        if (status != MB_OK) return status;
        output = std::move(runtime);
        return MB_OK;
    }
    constexpr std::array files{
        std::pair{"pose", "pose.gguf"}, std::pair{"root", "root.gguf"},
        std::pair{"vq-decoder", "vq-decoder.gguf"}, std::pair{"support", "support.gguf"},
    };
    for (const auto & [name, filename] : files) {
        const auto status = open_component(*runtime, bundle / filename, name, reason);
        if (status != MB_OK) return status;
    }
    output = std::move(runtime);
    return MB_OK;
}

mb_status create_neural_runtime(const std::filesystem::path & bundle, mb_device device,
    std::uint32_t threads, const std::string & directory,
    std::shared_ptr<neural_runtime> & output, std::string & reason) {
    return create_runtime(bundle, device, threads, directory, output, reason, false);
}

mb_status create_sonic_runtime(const std::filesystem::path & file, mb_device device,
    std::uint32_t threads, const std::string & directory,
    std::shared_ptr<neural_runtime> & output, std::string & reason) {
    return create_runtime(file, device, threads, directory, output, reason, true);
}

ggml_backend_t neural_backend(const neural_runtime & runtime) noexcept {
    return runtime.backend.get();
}

ggml_tensor * neural_weight(const neural_runtime & runtime,
                            std::string_view component, std::string_view name) noexcept {
    const auto found = std::find_if(runtime.components.begin(), runtime.components.end(),
        [&](const neural_runtime::component & item) { return item.name == component; });
    if (found == runtime.components.end()) return nullptr;
    std::string owned(name);
    if (auto * direct = ggml_get_tensor(found->context.get(), owned.c_str())) return direct;
    if (owned.size() >= 64U) {
        const std::string needle = "self_attn.";
        if (const auto position = owned.find(needle); position != std::string::npos)
            owned.replace(position, needle.size(), "attn.");
    }
    return ggml_get_tensor(found->context.get(), owned.c_str());
}

bool neural_copy_f32(const neural_runtime & runtime,
                     std::string_view component, std::string_view name,
                     std::vector<float> & output, std::string & reason) {
    auto * tensor = neural_weight(runtime, component, name);
    if (tensor == nullptr) {
        reason = "missing neural tensor: " + std::string(component) + ":" + std::string(name);
        return false;
    }
    if (tensor->type != GGML_TYPE_F32) {
        reason = "neural tensor is not F32: " + std::string(component) + ":" + std::string(name);
        return false;
    }
    output.resize(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    return true;
}

#else

class neural_runtime {};

mb_status create_sonic_runtime(const std::filesystem::path &, mb_device, std::uint32_t,
    const std::string &, std::shared_ptr<neural_runtime> &, std::string & reason) {
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
}

mb_status create_neural_runtime(const std::filesystem::path &, mb_device, std::uint32_t,
                                const std::string &, std::shared_ptr<neural_runtime> & output,
                                std::string & reason) {
    output.reset();
    reason = "this build has no GGML support";
    return MB_BACKEND_UNAVAILABLE;
}

bool neural_copy_f32(const neural_runtime &, std::string_view, std::string_view,
                     std::vector<float> &, std::string & reason) {
    reason = "this build has no GGML support";
    return false;
}

#endif

} // namespace motionbricks::detail
