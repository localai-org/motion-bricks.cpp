#include <motionbricks/sonic.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main(int argc,char ** argv) {
    assert(argc==3 || argc==4);
    char error[256]{};
    mb_runtime_options * options=nullptr;
    assert(mb_runtime_options_create(&options,error,sizeof(error))==MB_OK);
    if(argc==4) {
        assert(std::string(argv[3])=="vulkan");
        assert(mb_runtime_options_set_device(options,MB_DEVICE_VULKAN,error,sizeof(error))==MB_OK);
    }
    mb_sonic * combined=nullptr,* legacy=nullptr;
    assert(mb_sonic_load(argv[1],options,&combined,error,sizeof(error))==MB_OK);
    assert(mb_sonic_load(argv[2],options,&legacy,error,sizeof(error))==MB_OK);
    std::array<float,1762> obs{};
    std::array<float,64> tokens{},baseline{},old_tokens{};
    auto encode=[&](mb_sonic * m,auto & in,auto & out){
        return mb_sonic_encode(m,in.data(),in.size(),out.data(),out.size(),error,sizeof(error));
    };
    // Old files retain G1 results and reject mode 2 with no output writes.
    assert(encode(combined,obs,baseline)==MB_OK);
    assert(encode(legacy,obs,old_tokens)==MB_OK && baseline==old_tokens);
    obs[0]=2;
    tokens.fill(42);
    assert(encode(legacy,obs,tokens)==MB_INVALID_ARGUMENT && tokens[0]==42);
    assert(encode(combined,obs,tokens)==MB_OK && tokens!=baseline);
    const auto smpl_zero=tokens;
    // Unused robot fields cannot affect SMPL; all inputs are still validated.
    for(size_t i=1;i<922;++i) obs[i]=float(i)*.001F;
    assert(encode(combined,obs,tokens)==MB_OK && tokens==smpl_zero);
    obs[100]=std::numeric_limits<float>::quiet_NaN();
    assert(encode(combined,obs,tokens)==MB_INVALID_ARGUMENT);
    obs[100]=0;
    for(size_t index:{size_t(922),size_t(1642),size_t(1702),size_t(1761)}) {
        obs[index]=std::numeric_limits<float>::infinity();
        assert(encode(combined,obs,tokens)==MB_INVALID_ARGUMENT);
        obs[index]=std::numeric_limits<float>::max();
        assert(encode(combined,obs,tokens)==MB_INVALID_ARGUMENT);
        obs[index]=0;
    }
    for(float mode:{-1.0F,1.0F,3.0F,2.5F}) {
        obs[0]=mode;
        assert(encode(combined,obs,tokens)==MB_INVALID_ARGUMENT);
    }
    // Cache/trace isolation across modes, single and batched execution.
    constexpr uint32_t batch=4;
    std::vector<float> inputs(batch*1762),expected(batch*64),actual(batch*64);
    for(int mode:{2,0,2}) {
        for(uint32_t item=0;item<batch;++item) {
            auto * in=inputs.data()+item*1762;
            in[0]=float(mode);
            for(size_t j=1;j<1762;++j) in[j]=float(int((j+item*7)%37)-18)*.0001F;
            assert(mb_sonic_encode(combined,in,1762,expected.data()+item*64,64,error,sizeof(error))==MB_OK);
        }
        auto single_layers=std::vector<float>(2048);
        uint64_t n=0;
        assert(mb_sonic_layer(combined,1,0,single_layers.data(),single_layers.size(),&n,error,sizeof(error))==MB_OK && n==2048);
        assert(mb_sonic_encode_batch(combined,inputs.data(),inputs.size(),actual.data(),actual.size(),batch,error,sizeof(error))==MB_OK);
        assert(actual==expected);
        auto layers=std::vector<float>(batch*2048);
        assert(mb_sonic_layer(combined,1,0,layers.data(),layers.size(),&n,error,sizeof(error))==MB_OK && n==layers.size());
        for(size_t j=0;j<2048;++j) assert(std::abs(single_layers[j]-layers[(batch-1)*2048+j])<2e-5F);
        inputs[1762]=mode==2?0.0F:2.0F;
        std::fill(actual.begin(),actual.end(),42);
        assert(mb_sonic_encode_batch(combined,inputs.data(),inputs.size(),actual.data(),actual.size(),batch,error,sizeof(error))==MB_INVALID_ARGUMENT);
        assert(std::all_of(actual.begin(),actual.end(),[](float v){return v==42;}));
    }
    // Public maximum batch size uses a separate graph and preserves FSQ codes.
    inputs.assign(MB_SONIC_MAX_BATCH*1762,0);
    actual.resize(MB_SONIC_MAX_BATCH*64);
    for(uint32_t i=0;i<MB_SONIC_MAX_BATCH;++i) inputs[i*1762]=2;
    assert(mb_sonic_encode_batch(combined,inputs.data(),inputs.size(),actual.data(),actual.size(),MB_SONIC_MAX_BATCH,error,sizeof(error))==MB_OK);
    for(uint32_t i=0;i<MB_SONIC_MAX_BATCH;++i)
        assert(std::equal(smpl_zero.begin(),smpl_zero.end(),actual.begin()+i*64));
    mb_sonic_free(legacy);mb_sonic_free(combined);mb_runtime_options_free(options);
}
