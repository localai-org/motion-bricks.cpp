#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace motionbricks::detail {
// Diagnostic sm_120 instruction calibration, not model weights. Initialize
// once before starting graph workers; workers only read the immutable bytes.
inline std::vector<std::uint8_t> soma_rsqrt_corrections;
inline constexpr const char * soma_rsqrt_table_sha256=
    "002e5ef8ce6e36e6beec010a9e941a82c6a8ba997f273b3ab05fbe5d1eb86ba2";
inline std::mutex soma_rsqrt_table_mutex;
inline void load_soma_rsqrt_table(const char * path,bool reusable=false) {
    std::lock_guard lock(soma_rsqrt_table_mutex);
    if(!reusable && !soma_rsqrt_corrections.empty())throw std::runtime_error("rsqrt calibration already initialized");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=4194304)throw std::runtime_error("invalid rsqrt calibration size");
    std::vector<std::uint8_t> bytes(4194304);
    file.seekg(0);file.read(reinterpret_cast<char *>(bytes.data()),bytes.size());
    if(!file)throw std::runtime_error("cannot read rsqrt calibration");
    // Pinned content check; the full SHA256 is recorded with diagnostic output.
    std::uint64_t hash=14695981039346656037ULL;
    for(auto value:bytes)hash=(hash^value)*1099511628211ULL;
    if(hash!=0xba1c0fe6364a79d6ULL)throw std::runtime_error("rsqrt calibration checksum mismatch");
    // Never mutate a published table: concurrent model workers may be reading it.
    if(soma_rsqrt_corrections.empty())soma_rsqrt_corrections=std::move(bytes);
}
inline float soma_calibrated_rsqrt(float input) {
    const auto bits=std::bit_cast<std::uint32_t>(input);
    const int exponent=int((bits>>23)&255)-127;
    // The calibration covers positive normal inputs and normal outputs only.
    if((bits>>31) || exponent==-127 || exponent==128)
        throw std::runtime_error("rsqrt calibration requires positive normal input");
    const int shift=exponent>=0?exponent/2:(exponent-1)/2;
    const auto normalized_bits=(bits&0x7fffffU)|std::uint32_t(127+exponent-2*shift)<<23;
    const auto index=normalized_bits-0x3f800000U;
    const int delta=int((soma_rsqrt_corrections.at(index/4)>>(2*(index%4)))&3)-2;
    const float rounded=static_cast<float>(1.0/std::sqrt(double(std::bit_cast<float>(normalized_bits))));
    const auto corrected=std::bit_cast<float>(std::uint32_t(std::int64_t(std::bit_cast<std::uint32_t>(rounded))+delta));
    return std::ldexp(corrected,-shift);
}
inline const char * soma_welford_policy() {
    return soma_rsqrt_corrections.empty()?"welford-rsqrt-rn-fma-trial":"welford-sm120-rsqrt-table-fma-trial";
}
}
