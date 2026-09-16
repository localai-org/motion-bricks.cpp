#include "neural_runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <gguf.h>
#if defined(GGML_BACKEND_DL)
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
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

#if defined(GGML_BACKEND_DL)
std::filesystem::path backend_library_directory() {
    // Locate libggml itself, including when motionbricks is statically linked
    // or loaded by Python/Go from outside the installation directory.
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&ggml_backend_load_all), &module)) {
        std::vector<wchar_t> path(32768);
        const auto size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (size > 0 && size < path.size())
            return std::filesystem::path(std::wstring(path.data(), size)).parent_path();
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&ggml_backend_load_all), &info) && info.dli_fname)
        return std::filesystem::absolute(info.dli_fname).parent_path();
#endif
    return {};
}
#endif

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
    mb_device device = MB_DEVICE_CPU;
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

mb_status initialize_backend(neural_runtime & runtime, mb_device selected,
                             std::uint32_t threads, const std::string & directory,
                             std::string & reason) {
    // GGML's process-wide registry and backend discovery are not thread-safe.
    static std::mutex registry_mutex;
    const std::lock_guard lock(registry_mutex);
    try {
#if defined(MOTIONBRICKS_HAVE_VULKAN)
        // Set these before module discovery, which can initialize Vulkan.
        // Keep the released F32 model's parity contract on every backend.
#if defined(_WIN32)
        _putenv_s("GGML_VK_DISABLE_F16", "1");
        _putenv_s("GGML_VK_DISABLE_COOPMAT", "1");
        _putenv_s("GGML_VK_DISABLE_COOPMAT2", "1");
#else
        setenv("GGML_VK_DISABLE_F16", "1", 0);
        setenv("GGML_VK_DISABLE_COOPMAT", "1", 0);
        setenv("GGML_VK_DISABLE_COOPMAT2", "1", 0);
#endif
#endif
#if defined(GGML_BACKEND_DL)
        // Selection is process-wide. An explicit path overrides the bundled
        // location on first load; failed discovery remains retryable.
        static bool backends_loaded = false;
        if (!backends_loaded) {
            const auto search = directory.empty() ? backend_library_directory().string() : directory;
            if (!search.empty()) ggml_backend_load_all_from_path(search.c_str());
            backends_loaded = ggml_backend_reg_by_name("CPU") != nullptr;
        }
#else
        (void) directory;
#endif
        if (selected == MB_DEVICE_CPU) {
            runtime.backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            if (runtime.backend) {
                const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(runtime.backend.get()));
                const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
                if (!set_threads) {
                    reason = "CPU backend cannot configure inference threads";
                    return MB_BACKEND_UNAVAILABLE;
                }
                const auto hardware = std::max(1U, std::thread::hardware_concurrency());
                const auto count = threads == 0U ? hardware : threads;
                set_threads(runtime.backend.get(), static_cast<int>(count));
            }
        } else if (selected == MB_DEVICE_VULKAN) {
#if defined(MOTIONBRICKS_HAVE_VULKAN)
            const auto reg = ggml_backend_reg_by_name("Vulkan");
            if (reg && ggml_backend_reg_dev_count(reg) > 0)
                runtime.backend.reset(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr));
#else
            reason = "this build has no Vulkan backend";
            return MB_BACKEND_UNAVAILABLE;
#endif
        }
    } catch (const std::exception & e) {
        reason = std::string("cannot initialize inference backend: ") + e.what();
        return MB_BACKEND_UNAVAILABLE;
    }
    if (!runtime.backend) {
        reason = selected == MB_DEVICE_VULKAN ? "no accessible Vulkan devices/backend"
                                             : "cannot find or initialize a compatible CPU backend";
        return MB_BACKEND_UNAVAILABLE;
    }
    return MB_OK;
}

} // namespace

static mb_status create_runtime(const std::filesystem::path & bundle,
                                mb_device device, std::uint32_t threads,
                                const std::string & backend_directory,
                                std::shared_ptr<neural_runtime> & output,
                                std::string & reason, bool sonic) {
    output.reset();
    auto runtime = std::make_shared<neural_runtime>();
    const auto selected = device == MB_DEVICE_AUTO ? MB_DEVICE_CPU : device;
    runtime->device = selected;
    const auto initialized = initialize_backend(*runtime, selected, threads, backend_directory, reason);
    if (initialized != MB_OK) return initialized;
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

mb_device neural_device(const neural_runtime & runtime) noexcept { return runtime.device; }

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

mb_device neural_device(const neural_runtime &) noexcept { return MB_DEVICE_CPU; }

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
