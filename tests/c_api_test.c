#include <motionbricks/motionbricks.h>

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    char error[128] = {0};
    mb_runtime_options * options = 0;
    mb_command * command = 0;

    assert(mb_abi_version() == MB_ABI_VERSION);
    assert(strcmp(mb_status_string(MB_OK), "ok") == 0);
    assert(mb_runtime_options_create(&options, error, sizeof(error)) == MB_OK);
    assert(options != 0 && error[0] == '\0');

    mb_device device = UINT32_MAX;
    uint32_t threads = UINT32_MAX;
    const char * directory = 0;
    assert(mb_runtime_options_get_device(options, &device, error, sizeof(error)) == MB_OK);
    assert(device == MB_DEVICE_AUTO);
    assert(mb_runtime_options_get_threads(options, &threads, error, sizeof(error)) == MB_OK);
    assert(threads == 0);
    assert(mb_runtime_options_set_device(options, MB_DEVICE_CPU, error, sizeof(error)) == MB_OK);
    assert(mb_runtime_options_set_threads(options, 7, error, sizeof(error)) == MB_OK);
    assert(mb_runtime_options_set_backend_directory(options, "/tmp/backends", error, sizeof(error)) == MB_OK);
    assert(mb_runtime_options_get_backend_directory(options, &directory, error, sizeof(error)) == MB_OK);
    assert(strcmp(directory, "/tmp/backends") == 0);
    assert(mb_runtime_options_set_device(options, UINT32_C(999), error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(error[0] != '\0');

    assert(mb_command_create(&command, error, sizeof(error)) == MB_OK);
    assert(command != 0);
    uint32_t sampling_argmax = 99;
    assert(mb_command_get_sampling_argmax(command, &sampling_argmax, error, sizeof(error)) == MB_OK && sampling_argmax == 0);
    assert(mb_command_set_sampling_argmax(command, 1, error, sizeof(error)) == MB_OK);
    assert(mb_command_set_sampling_argmax(command, 2, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_command_get_sampling_argmax(command, &sampling_argmax, error, sizeof(error)) == MB_OK && sampling_argmax == 1);
    assert(mb_command_set_sampling_argmax(0, 0, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_command_get_sampling_argmax(command, 0, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    float x = 0.0F, y = 0.0F, z = 0.0F, speed = 0.0F, heading = 0.0F;
    uint32_t enabled = UINT32_MAX;
    uint64_t seed = UINT64_MAX;
    assert(mb_command_get_facing_direction(command, &x, &y, &z, error, sizeof(error)) == MB_OK);
    assert(x == 0.0F && y == 0.0F && z == 1.0F);
    assert(mb_command_get_target_speed(command, &speed, error, sizeof(error)) == MB_OK);
    assert(speed == -1.0F);
    assert(mb_command_set_movement_direction(command, 1.0F, 0.0F, -0.5F, error, sizeof(error)) == MB_OK);
    assert(mb_command_get_movement_direction(command, &x, &y, &z, error, sizeof(error)) == MB_OK);
    assert(x == 1.0F && y == 0.0F && z == -0.5F);
    assert(mb_command_set_facing_direction(command, 0.0F, 0.0F, 0.0F, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_command_set_movement_direction(command, NAN, 0.0F, 0.0F, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_command_set_target_speed(command, -1.01F, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_command_set_world_target(command, 2.0F, 1.0F, 3.0F, 0.25F, 1, error, sizeof(error)) == MB_OK);
    assert(mb_command_get_world_target(command, &x, &y, &z, &heading, &enabled, error, sizeof(error)) == MB_OK);
    assert(x == 2.0F && y == 1.0F && z == 3.0F && heading == 0.25F && enabled == 1);
    assert(mb_command_set_seed(command, UINT64_C(42), error, sizeof(error)) == MB_OK);
    assert(mb_command_get_seed(command, &seed, error, sizeof(error)) == MB_OK && seed == UINT64_C(42));

    mb_model * model = (mb_model *)(uintptr_t)1;
    assert(mb_model_load("fixture", options, &model, error, sizeof(error)) == MB_IO_ERROR);
    assert(model == 0 && error[0] != '\0');
    assert(mb_model_get_parameter_count(0, &seed, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_model_get_neutral_joint_position(0, 0, &x, &y, &z, error, sizeof(error)) == MB_INVALID_ARGUMENT);
    assert(mb_motion_get_target_frame_count(0, &seed, error, sizeof(error)) == MB_INVALID_ARGUMENT);

    {
        char tiny[2] = {'x', 'x'};
        assert(mb_runtime_options_set_device(options, UINT32_C(999), tiny, sizeof(tiny)) == MB_INVALID_ARGUMENT);
        assert(tiny[1] == '\0');
    }

    mb_command_free(command);
    mb_runtime_options_free(options);
    mb_command_free(0);
    mb_runtime_options_free(0);
    return 0;
}
