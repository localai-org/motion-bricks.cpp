#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace motionbricks::detail {
// Diagnostic calibration of sm_120 EX2, not a portable CUDA implementation.
// Initialize before graph workers start. The table covers [2^-24,1) only;
// other reduced arguments use double exp2 rounded to float (not CUDA-exact).
inline std::vector<std::uint8_t> soma_exp2_corrections;
inline constexpr const char * soma_exp2_table_sha256=
    "11d89bff841e859226e57a64735f294cd2cc2b77665f25b2b953ea3ff83fcbc5";
inline std::mutex soma_exp2_table_mutex;
inline void load_soma_exp2_table(const char * path,bool reusable=false) {
    std::lock_guard lock(soma_exp2_table_mutex);
    if(!reusable && !soma_exp2_corrections.empty())throw std::runtime_error("exp2 calibration already initialized");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=50331648)throw std::runtime_error("invalid exp2 calibration size");
    std::vector<std::uint8_t> bytes(50331648);
    file.seekg(0);file.read(reinterpret_cast<char *>(bytes.data()),bytes.size());
    if(!file)throw std::runtime_error("cannot read exp2 calibration");
    std::uint64_t hash=14695981039346656037ULL;
    for(auto value:bytes)hash=(hash^value)*1099511628211ULL;
    if(hash!=0xd624a1b99212d232ULL)throw std::runtime_error("exp2 calibration checksum mismatch");
    // Never mutate a published table: concurrent model workers may be reading it.
    if(soma_exp2_corrections.empty())soma_exp2_corrections=std::move(bytes);
}
inline float soma_softmax_exp(float x) {
    if(x==-INFINITY)return 0;
    if(!std::isfinite(x) || x>0)throw std::runtime_error("softmax exponential requires nonpositive input");
    const float sat=std::clamp(std::fma(x,0.005724980030208826F,0.5F),0.0F,1.0F);
    // The range-reduction FFMA.RM result has unit spacing in this interval.
    const int n=static_cast<int>(std::floor(double(sat)*252.0+12582913.0))-12583039;
    float reduced=std::fma(x,1.4426950216293334961F,-float(n));
    reduced=std::fma(x,1.925963033500011079e-08F,reduced);
    float value=static_cast<float>(std::exp2(double(reduced)));
    const auto bits=std::bit_cast<std::uint32_t>(reduced);
    if(bits>=0x33800000U && bits<0x3f800000U) {
        const auto index=bits-0x33800000U;
        const int delta=int((soma_exp2_corrections.at(index/4)>>(2*(index%4)))&3)-2;
        value=std::bit_cast<float>(std::uint32_t(std::int64_t(std::bit_cast<std::uint32_t>(value))+delta));
    }
    return std::ldexp(value,n);
}
inline void soma_warp_softmax(const float * input,float * output,std::size_t width) {
    if(!width || width>1024)throw std::runtime_error("unsupported diagnostic softmax width");
    const float maximum=*std::max_element(input,input+width);
    if(!std::isfinite(maximum))throw std::runtime_error("diagnostic softmax needs a finite row maximum");
    const auto warp=std::min<std::size_t>(32,std::bit_ceil(width));
    std::array<float,32> sums{};
    for(std::size_t lane=0;lane<warp;++lane)
        for(std::size_t i=lane;i<width;i+=warp) {
            output[i]=soma_softmax_exp(input[i]-maximum);
            sums[lane]+=output[i];
        }
    // Lane zero of the source XOR shuffle tree; other lanes have the same sum.
    for(auto offset=warp/2;offset;offset/=2)
        for(std::size_t i=0;i<offset;++i)sums[i]+=sums[i+offset];
    for(std::size_t i=0;i<width;++i)output[i]/=sums[0];
}
}
