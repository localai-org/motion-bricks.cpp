#include "sampling.hpp"
#include <motionbricks/motionbricks.h>
#include <algorithm>
#include <cassert>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t * data, std::size_t size) {
    if (size < 4 || size > 16384) return 0;
    std::uint32_t mode=0;std::memcpy(&mode,data,4);
    char error[64]{};mb_command * command=nullptr;
    if (mb_command_create(&command,error,sizeof error)!=MB_OK) return 0;
    const auto status=mb_command_set_sampling_argmax(command,mode,error,sizeof error);
    assert(status==(mode<=1 ? MB_OK : MB_INVALID_ARGUMENT));
    std::uint32_t readback=99;
    assert(mb_command_get_sampling_argmax(command,&readback,error,sizeof error)==MB_OK);
    assert(readback==(mode<=1 ? mode : 0));
    mb_command_free(command);
    const auto count=(size-4)/8;
    std::vector<float> logits(count),uniforms(count);
    for(std::size_t i=0;i<count;++i) {
        std::memcpy(&logits[i],data+4+i*8,4);
        std::memcpy(&uniforms[i],data+8+i*8,4);
        if(mode&1) uniforms[i]=static_cast<float>(data[8+i*8])/255.0F;
    }
    std::vector<std::int32_t> selected,repeat;
    std::string reason;
    const auto choices=mode%12;
    const auto input=(mode&2) ? std::span<const float>{} : std::span<const float>(uniforms);
    const bool ok=motionbricks::detail::sample_pose_tokens(logits,input,choices,selected,reason);
    if(ok) {
        assert(selected.size()==count/choices);
        for(auto value:selected)assert(value>=0 && static_cast<std::uint32_t>(value)<choices);
        assert(motionbricks::detail::sample_pose_tokens(logits,input,choices,repeat,reason));
        assert(selected==repeat);
    }
    return 0;
}
