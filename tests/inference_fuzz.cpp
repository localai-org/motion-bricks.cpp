// No GGUF/parser fuzzing. An optional trusted model is loaded once.
#include <motionbricks/inference.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t * bytes,size_t size) {
    if(size<8)return 0;
    char e[64];mb_inference_request * r=nullptr;
    if(mb_inference_request_create(&r,e,sizeof e)!=MB_OK)return 0;
    std::array<float,2424> values{},out{};
    std::memcpy(values.data(),bytes+8,std::min(size-8,sizeof(values)));
    std::array<uint32_t,11> flags{},readback{};
    for(size_t i=0;i<11;++i)flags[i]=bytes[i%size]%(bytes[0]&1?2:4);
    uint64_t count=0;uint32_t mode=0;
    mb_inference_request_set_seed(r,bytes[0],e,sizeof e);
    mb_inference_request_get_seed(r,&count,e,sizeof e);
    mb_inference_request_set_sampling_argmax(r,bytes[1],e,sizeof e);
    mb_inference_request_get_sampling_argmax(r,&mode,e,sizeof e);
    for(uint32_t field=0;field<5;++field) {
        const uint64_t wanted=field==0?40:field==1?32:2424;
        mb_inference_request_set_features(r,field,bytes[2]&1?values.data():nullptr,bytes[3]&1?wanted:bytes[3],e,sizeof e);
        mb_inference_request_get_features(r,field,bytes[4]&1?out.data():nullptr,bytes[5]&1?2424:0,&count,e,sizeof e);
        mb_inference_request_set_mask(r,field,flags.data(),bytes[6]&1?(field==3?11:8):0,e,sizeof e);
        mb_inference_request_get_mask(r,field,readback.data(),bytes[7]%12,&count,e,sizeof e);
    }
    static mb_model * model=[](){mb_model * m=nullptr;char error[128];
        const char * path=std::getenv("MOTIONBRICKS_FUZZ_BUNDLE");
        mb_runtime_options * options=nullptr;
        mb_runtime_options_create(&options,error,sizeof error);
        mb_runtime_options_set_device(options,MB_DEVICE_CPU,error,sizeof error);
        mb_runtime_options_set_threads(options,2,error,sizeof error);
        if(path&&mb_model_load(path,options,&m,error,sizeof error)!=MB_OK)std::abort();
        mb_runtime_options_free(options);return m;}();
    mb_inference_request_set_boundary_poses(r,model,bytes[0]%3,values.data(),bytes[1]&1?12:0,values.data()+12,bytes[2]&1?544:0,e,sizeof e);
    if(model&&(bytes[0]&15)==0) {
        // Reach successful conversion/inference as well as rejection branches.
        // Inputs remain canonical and small; mutate one quaternion when asked.
        std::array<float,12> roots{};std::array<float,544> rotations{};
        for(size_t f=0;f<4;++f) {
            roots[f*3+1]=.8F;roots[f*3+2]=float(f)*float(bytes[1])*.0001F;
            for(size_t j=0;j<34;++j)rotations[(f*34+j)*4+3]=1;
        }
        if(bytes[2]&1)rotations[bytes[3]%136]=values[0];
        for(uint32_t b=0;b<2;++b)mb_inference_request_set_boundary_poses(r,model,b,roots.data(),12,rotations.data(),544,e,sizeof e);
    }
    mb_motion * motion=nullptr;
    mb_model_infer(model,r,&motion,e,sizeof e);
    mb_motion_free(motion);mb_inference_request_free(r);return 0;
}
