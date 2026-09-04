#pragma once

#include <motionbricks/motionbricks.h>

#include <cstdint>
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

mb_status create_neural_runtime(const std::filesystem::path & bundle,
                                mb_device device, std::uint32_t threads,
                                const std::string & backend_directory,
                                std::shared_ptr<neural_runtime> & output,
                                std::string & reason);

#if defined(MOTIONBRICKS_HAVE_GGML)
ggml_backend_t neural_backend(const neural_runtime & runtime) noexcept;
ggml_tensor * neural_weight(const neural_runtime & runtime,
                            std::string_view component, std::string_view name) noexcept;
#endif
bool neural_copy_f32(const neural_runtime & runtime,
                     std::string_view component, std::string_view name,
                     std::vector<float> & output, std::string & reason);

} // namespace motionbricks::detail
