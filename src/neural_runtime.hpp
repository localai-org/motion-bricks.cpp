#pragma once

#include <motionbricks/motionbricks.h>

#include <cstdint>
#include <cstddef>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml-backend.h>
#include <ggml.h>
#endif

namespace motionbricks::detail {

class neural_runtime;

// Debug-only snapshots in GGML dimension order, packed without view strides.
// Keeping the original element type preserves exact integer and FP16 values.
struct neural_tensor_trace {
    std::string name;
    std::string operation;
    std::array<std::int64_t, 4> shape;
    int type;
    std::vector<std::byte> data;
};

class neural_trace_scope {
public:
    neural_trace_scope(const neural_runtime &, std::vector<neural_tensor_trace> &, std::string prefix);
    ~neural_trace_scope();
    neural_trace_scope(const neural_trace_scope &) = delete;
    neural_trace_scope & operator=(const neural_trace_scope &) = delete;
private:
    const neural_runtime & runtime_;
    std::vector<neural_tensor_trace> * previous_;
    std::string previous_prefix_;
};

mb_status create_sonic_runtime(const std::filesystem::path & file,
                              mb_device device, std::uint32_t threads,
                              const std::string & backend_directory,
                              std::shared_ptr<neural_runtime> & output,
                              std::string & reason);

mb_status create_neural_runtime(const std::filesystem::path & bundle,
                                mb_device device, std::uint32_t threads,
                                const std::string & backend_directory,
                                std::shared_ptr<neural_runtime> & output,
                                std::string & reason);

#if defined(MOTIONBRICKS_HAVE_GGML)
ggml_backend_t neural_backend(const neural_runtime & runtime) noexcept;
ggml_status neural_graph_compute(const neural_runtime & runtime, ggml_cgraph * graph);
ggml_tensor * neural_weight(const neural_runtime & runtime,
                            std::string_view component, std::string_view name) noexcept;
#endif
mb_device neural_device(const neural_runtime & runtime) noexcept;
// Qualification is deliberately limited to the captured device/driver and
// loaded overlay protocol. A successful build alone does not establish parity.
bool neural_soma_vulkan_sm120(const neural_runtime & runtime, std::string & reason);
bool neural_copy_f32(const neural_runtime & runtime,
                     std::string_view component, std::string_view name,
                     std::vector<float> & output, std::string & reason);

} // namespace motionbricks::detail
