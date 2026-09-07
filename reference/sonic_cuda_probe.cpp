// Minimal diagnostic to distinguish CUDA setup failures from SONIC inference.
#include <cuda_runtime_api.h>
#include <cstdio>

static bool check(const char * call, cudaError_t status) {
    std::printf("%s: %d (%s)\n", call, int(status), cudaGetErrorString(status));
    return status == cudaSuccess;
}

int main() {
    int driver = 0, runtime = 0, count = 0;
    check("cudaDriverGetVersion", cudaDriverGetVersion(&driver));
    check("cudaRuntimeGetVersion", cudaRuntimeGetVersion(&runtime));
    std::printf("driver=%d runtime=%d\n", driver, runtime);
    if (!check("cudaGetDeviceCount", cudaGetDeviceCount(&count))) return 1;
    std::printf("devices=%d\n", count);
    if (!check("cudaSetDevice", cudaSetDevice(0))) return 1;
    if (!check("cudaFree(0)", cudaFree(nullptr))) return 1;
    size_t free = 0, total = 0;
    if (!check("cudaMemGetInfo", cudaMemGetInfo(&free, &total))) return 1;
    std::printf("free=%zu total=%zu\n", free, total);
}
