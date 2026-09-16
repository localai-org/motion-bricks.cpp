#ifndef MOTIONBRICKS_MOTIONBRICKS_H
#define MOTIONBRICKS_MOTIONBRICKS_H

#include <stdint.h>

#define MB_ABI_VERSION UINT32_C(1)

#if defined(_WIN32) && defined(MOTIONBRICKS_SHARED)
#  if defined(MOTIONBRICKS_BUILD)
#    define MB_API __declspec(dllexport)
#  else
#    define MB_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(MOTIONBRICKS_SHARED)
#  define MB_API __attribute__((visibility("default")))
#else
#  define MB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t mb_status;
#define MB_OK                   UINT32_C(0)
#define MB_INVALID_ARGUMENT     UINT32_C(1)
#define MB_OUT_OF_MEMORY        UINT32_C(2)
#define MB_IO_ERROR             UINT32_C(3)
#define MB_INVALID_FORMAT       UINT32_C(4)
#define MB_INCOMPATIBLE_MODEL   UINT32_C(5)
#define MB_BACKEND_UNAVAILABLE  UINT32_C(6)
#define MB_COMPUTE_FAILED       UINT32_C(7)
#define MB_NOT_IMPLEMENTED      UINT32_C(8)
#define MB_INTERNAL_ERROR       UINT32_C(9)

typedef uint32_t mb_device;
#define MB_DEVICE_AUTO    UINT32_C(0)
#define MB_DEVICE_CPU     UINT32_C(1)
#define MB_DEVICE_VULKAN  UINT32_C(2)

typedef struct mb_runtime_options mb_runtime_options;
typedef struct mb_model mb_model;
typedef struct mb_style mb_style;
typedef struct mb_command mb_command;
typedef struct mb_agent mb_agent;
typedef struct mb_motion mb_motion;

MB_API uint32_t mb_abi_version(void);
MB_API const char * mb_status_string(mb_status status);

/* Runtime options are opaque so FFI callers never reproduce a struct layout. */
MB_API mb_status mb_runtime_options_create(mb_runtime_options ** output,
                                           char * error, uint64_t error_capacity);
MB_API void mb_runtime_options_free(mb_runtime_options * value);
MB_API mb_status mb_runtime_options_set_device(mb_runtime_options * value, mb_device device,
                                               char * error, uint64_t error_capacity);
MB_API mb_status mb_runtime_options_get_device(const mb_runtime_options * value, mb_device * output,
                                               char * error, uint64_t error_capacity);
MB_API mb_status mb_runtime_options_set_threads(mb_runtime_options * value, uint32_t threads,
                                                char * error, uint64_t error_capacity);
MB_API mb_status mb_runtime_options_get_threads(const mb_runtime_options * value, uint32_t * output,
                                                char * error, uint64_t error_capacity);
/* Overrides backend discovery before the first model load in this process.
 * Passing NULL clears the override. The returned string is borrowed.
 * Successfully loaded backends are shared by subsequent model loads. */
MB_API mb_status mb_runtime_options_set_backend_directory(mb_runtime_options * value,
                                                          const char * directory,
                                                          char * error, uint64_t error_capacity);
MB_API mb_status mb_runtime_options_get_backend_directory(const mb_runtime_options * value,
                                                          const char ** output,
                                                          char * error, uint64_t error_capacity);

/* A command is independent of compiler layout and may be reused between plans. */
MB_API mb_status mb_command_create(mb_command ** output,
                                   char * error, uint64_t error_capacity);
MB_API void mb_command_free(mb_command * value);
MB_API mb_status mb_command_set_style(mb_command * value, const mb_style * style,
                                      char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_style(const mb_command * value, const mb_style ** output,
                                      char * error, uint64_t error_capacity);
MB_API mb_status mb_command_set_movement_direction(mb_command * value, float x, float y, float z,
                                                   char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_movement_direction(const mb_command * value,
                                                   float * x, float * y, float * z,
                                                   char * error, uint64_t error_capacity);
MB_API mb_status mb_command_set_facing_direction(mb_command * value, float x, float y, float z,
                                                 char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_facing_direction(const mb_command * value,
                                                 float * x, float * y, float * z,
                                                 char * error, uint64_t error_capacity);
/* A negative speed selects the style's configured speed. */
MB_API mb_status mb_command_set_target_speed(mb_command * value, float metres_per_second,
                                             char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_target_speed(const mb_command * value, float * output,
                                             char * error, uint64_t error_capacity);
MB_API mb_status mb_command_set_world_target(mb_command * value,
                                             float x, float y, float z,
                                             float heading_radians, uint32_t enabled,
                                             char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_world_target(const mb_command * value,
                                             float * x, float * y, float * z,
                                             float * heading_radians, uint32_t * enabled,
                                             char * error, uint64_t error_capacity);
MB_API mb_status mb_command_set_seed(mb_command * value, uint64_t seed,
                                    char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_seed(const mb_command * value, uint64_t * output,
                                    char * error, uint64_t error_capacity);
/* Default 0: upstream Gumbel sampling at temperature 1. Set 1 for diagnostic
   argmax. Each plan's private RNG is reset from command.seed; equal inputs and
   seed reproduce the same draws. PyTorch's seeded stream is not identical. */
MB_API mb_status mb_command_set_sampling_argmax(mb_command * value, uint32_t enabled,
                                               char * error, uint64_t error_capacity);
MB_API mb_status mb_command_get_sampling_argmax(const mb_command * value, uint32_t * output,
                                               char * error, uint64_t error_capacity);

/* Neural model, style, and stateful animation planner. */
MB_API mb_status mb_model_load(const char * bundle_directory,
                               const mb_runtime_options * options, mb_model ** output,
                               char * error, uint64_t error_capacity);
MB_API void mb_model_free(mb_model * value);
MB_API mb_status mb_model_get_parameter_count(const mb_model * value, uint64_t * output,
                                              char * error, uint64_t error_capacity);
MB_API mb_status mb_model_get_joint_count(const mb_model * value, uint32_t * output,
                                          char * error, uint64_t error_capacity);
MB_API mb_status mb_model_get_joint_name(const mb_model * value, uint32_t joint,
                                         const char ** output,
                                         char * error, uint64_t error_capacity);
MB_API mb_status mb_model_get_joint_parent(const mb_model * value, uint32_t joint,
                                           int32_t * output,
                                           char * error, uint64_t error_capacity);
MB_API mb_status mb_model_get_neutral_joint_position(const mb_model * value, uint32_t joint,
                                                     float * x, float * y, float * z,
                                                     char * error, uint64_t error_capacity);

MB_API mb_status mb_style_load(const mb_model * model, const char * style_path,
                               mb_style ** output, char * error, uint64_t error_capacity);
MB_API void mb_style_free(mb_style * value);
MB_API mb_status mb_style_get_name(const mb_style * value, const char ** output,
                                   char * error, uint64_t error_capacity);
MB_API mb_status mb_style_set_speed(mb_style * value, float metres_per_second,
                                    char * error, uint64_t error_capacity);
MB_API mb_status mb_style_get_speed(const mb_style * value, float * output,
                                    char * error, uint64_t error_capacity);

MB_API mb_status mb_agent_create(const mb_model * model, mb_agent ** output,
                                 char * error, uint64_t error_capacity);
MB_API void mb_agent_free(mb_agent * value);
MB_API mb_status mb_agent_reset(mb_agent * value, const mb_style * initial_style,
                                char * error, uint64_t error_capacity);
MB_API mb_status mb_agent_set_context(mb_agent * value,
                                      const float * root_xyz,
                                      const float * local_rotation_xyzw,
                                      uint64_t frames, uint64_t joints,
                                      char * error, uint64_t error_capacity);
MB_API mb_status mb_agent_plan(mb_agent * value, const mb_command * command,
                               mb_motion ** output,
                               char * error, uint64_t error_capacity);
/* Plans count independent robots through one shared model call. Every
   agent must belong to the same model and occur only once. Commands and
   outputs are parallel arrays; outputs are all NULL on failure. Count 1..64. */
MB_API mb_status mb_agent_plan_batch(mb_agent * const * agents,
                                     const mb_command * const * commands,
                                     uint64_t count, mb_motion ** outputs,
                                     char * error, uint64_t error_capacity);
MB_API mb_status mb_agent_advance(mb_agent * value, uint32_t frames,
                                  char * error, uint64_t error_capacity);

MB_API void mb_motion_free(mb_motion * value);
MB_API mb_status mb_motion_get_frame_count(const mb_motion * value, uint64_t * output,
                                           char * error, uint64_t error_capacity);
MB_API mb_status mb_motion_get_joint_count(const mb_motion * value, uint64_t * output,
                                           char * error, uint64_t error_capacity);
/* Borrowed row-major F32 buffers valid until mb_motion_free. */
MB_API mb_status mb_motion_get_root_translations(const mb_motion * value,
                                                 const float ** output, uint64_t * values,
                                                 char * error, uint64_t error_capacity);
MB_API mb_status mb_motion_get_local_rotations_xyzw(const mb_motion * value,
                                                    const float ** output, uint64_t * values,
                                                    char * error, uint64_t error_capacity);
/* The four placed style-pose constraints actually supplied to the planner. */
MB_API mb_status mb_motion_get_target_frame_count(const mb_motion * value, uint64_t * output,
                                                  char * error, uint64_t error_capacity);
MB_API mb_status mb_motion_get_target_root_translations(const mb_motion * value,
                                                        const float ** output, uint64_t * values,
                                                        char * error, uint64_t error_capacity);
MB_API mb_status mb_motion_get_target_local_rotations_xyzw(const mb_motion * value,
                                                           const float ** output, uint64_t * values,
                                                           char * error, uint64_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
