// Private, versioned capability query for the MotionBricks GGML overlay.
// Keep the wire format independent of Vulkan headers and C++ object layouts.
// This query deliberately does not initialize a logical device. The host can
// decide its capability policy before ggml_vk_get_device fixes those settings.
static bool motionbricks_soma_vulkan_can_enable_v1(ggml_backend_dev_t dev) {
    if (!dev) return false;
    auto * ctx = static_cast<ggml_backend_vk_device_context *>(dev->context);
    const auto devices = vk_instance.instance.enumeratePhysicalDevices();
    const auto physical = devices.at(vk_instance.device_indices.at(ctx->device));
    const auto props = physical.getProperties();
    return props.vendorID == 0x10de && props.deviceID == 0x2c05 && props.driverVersion == 0x94d1c140;
}

static bool motionbricks_soma_vulkan_info_v1(ggml_backend_dev_t dev, uint32_t * values, size_t count) {
    if (!dev || !values || count != 8) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_vk_device_context *>(dev->context);
    const auto device = ggml_vk_get_device(ctx->device);
    values[0] = 1; // protocol version: scoped norm/softmax and FP32 coexistence
    values[1] = device->properties.vendorID;
    values[2] = device->properties.deviceID;
    values[3] = device->properties.driverVersion;
    values[4] = device->fp16 ? 1 : 0;
    values[5] = device->coopmat2 ? 1 : 0;
    values[6] = device->subgroup_size;
    values[7] = static_cast<uint32_t>(device->driver_id);
    return true;
}

static void * motionbricks_soma_vulkan_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (name && strcmp(name, "motionbricks_soma_vulkan_can_enable_v1") == 0) {
        return reinterpret_cast<void *>(motionbricks_soma_vulkan_can_enable_v1);
    }
    if (name && strcmp(name, "motionbricks_soma_vulkan_info_v1") == 0) {
        return reinterpret_cast<void *>(motionbricks_soma_vulkan_info_v1);
    }
    return nullptr;
}
