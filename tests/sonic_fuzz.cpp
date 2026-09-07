// Model is loaded once from an explicitly supplied trusted local GGUF.
// Fuzz only inference/ABI input boundaries, never GGUF loading.
#include <motionbricks/physics.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * bytes,size_t size) {
    static mb_sonic * model=[](){mb_sonic * result=nullptr;char e[256]{};
        const char * path=std::getenv("MOTIONBRICKS_FUZZ_SONIC");
        if(path && mb_sonic_load(path,nullptr,&result,e,sizeof(e))!=MB_OK){std::fprintf(stderr,"%s\n",e);std::abort();}return result;}();
    if(size<4)return 0;
    std::array<float,1762> input{};std::array<float,64> output{};char error[64]{};
    std::memcpy(input.data(),bytes+4,std::min(size-4,input.size()*sizeof(float)));
    input[0]=bytes[0]%3;
    if(bytes[1]&1) mb_sonic_encode(model,input.data(),bytes[2]&1?1762:bytes[2],output.data(),bytes[3]&1?64:bytes[3],error,sizeof(error));
    else mb_sonic_decode(model,input.data(),bytes[2]&1?994:bytes[2],output.data(),bytes[3]&1?29:bytes[3],error,sizeof(error));
    uint64_t count=0;mb_sonic_layer(model,bytes[0]%4,bytes[1]%9,output.data(),64,&count,error,sizeof(error));
    // Optional trusted scene/config loaded once, outside the fuzz surface.
    // Pose/count/time/trace arguments are fuzzed; model and XML loading aren't.
    static mb_physics * physics=[&](){mb_physics * result=nullptr;
        const char * scene=std::getenv("MOTIONBRICKS_FUZZ_SCENE"),*config=std::getenv("MOTIONBRICKS_FUZZ_CONFIG");
        if(scene && config && mb_physics_create(model,scene,config,&result,error,sizeof(error))!=MB_OK) std::abort();
        return result;}();
    std::array<float,90> actual{},reference{};
    if(physics) {
        uint32_t shapes=0,type=0;float size3[3]{};char name[256]{};
        mb_physics_collision_count(physics,&shapes,error,sizeof(error));
        const uint32_t shape=bytes[0]&4?shapes:(shapes?bytes[2]%shapes:0);
        mb_physics_collision_shape(physics,shape,&type,size3,bytes[1]&1?3:0,name,bytes[3],error,sizeof(error));
        mb_physics_collision_triangles(physics,shape,bytes[1]&1?output.data():nullptr,bytes[1]&1?bytes[3]%65:0,&count,error,sizeof(error));
        std::array<float,1024*7> collision_transforms{};
        if(shapes<=1024) mb_physics_collision_transforms(physics,collision_transforms.data(),bytes[1]&1?shapes*7:bytes[3],error,sizeof(error));
        if(bytes[0]&1) mb_physics_reset(physics,error,sizeof(error));
        mb_physics_start(physics,input.data(),bytes[2]&1?3:bytes[2],input.data()+8,bytes[3]&1?136:bytes[3],error,sizeof(error));
        mb_physics_step(physics,input.data(),6,input.data()+8,272,bytes[2]&1?2:bytes[2],double(input[5]),actual.data(),90,reference.data(),90,error,sizeof(error));
        std::array<double,1762> trace{};
        mb_physics_trace(physics,bytes[1]%8,trace.data(),bytes[3],&count,error,sizeof(error));
    }
    return 0;
}
