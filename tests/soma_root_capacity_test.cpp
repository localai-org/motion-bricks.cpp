#include "neural_runtime.hpp"
#include "neural_precision.hpp"
#include "root.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace motionbricks::detail;
struct fixture {
    ggml_context * tensors=nullptr;
    gguf_context * metadata=nullptr;
    explicit fixture(const char * path) {
        metadata=gguf_init_from_file(path,{false,&tensors});
        if(!metadata||!tensors)throw std::runtime_error("invalid fixture");
    }
    ~fixture(){gguf_free(metadata);ggml_free(tensors);}
    template<class T> std::vector<T> read(const char * name) const {
        auto * t=ggml_get_tensor(tensors,name);
        if(!t||ggml_element_size(t)!=sizeof(T))throw std::runtime_error(name);
        std::vector<T> out(ggml_nelements(t));std::memcpy(out.data(),t->data,ggml_nbytes(t));return out;
    }
};
int main(int argc,char ** argv) try {
    if(argc!=5)throw std::runtime_error("usage: soma-root-capacity BUNDLE FIXTURE BATCH fp32|mixed-fp16");
    const auto batch=static_cast<unsigned>(std::stoul(argv[3]));
    if(!batch||batch>64)throw std::runtime_error("invalid batch size");
    const std::string precision=argv[4];
    if(precision!="fp32"&&precision!="mixed-fp16")throw std::runtime_error("invalid precision");
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],MB_DEVICE_CPU,1,"",runtime,reason)!=MB_OK)throw std::runtime_error(reason);
    neural_precision_scope scope(precision=="mixed-fp16");fixture f(argv[2]);
    const auto floats=[&](const char * name){auto v=f.read<float>(name);std::vector<float> out;
        for(unsigned i=0;i<batch;++i)out.insert(out.end(),v.begin(),v.end());return out;};
    const auto masks=[&](const char * name){auto v=f.read<int>(name);std::vector<std::uint8_t> out;
        for(unsigned i=0;i<batch;++i)out.insert(out.end(),v.begin(),v.end());return out;};
    const auto global=floats("root.global"),local=floats("root.local"),poses=floats("root.poses");
    const auto gm=masks("root.global_mask"),lm=masks("root.local_mask"),pm=masks("root.pose_mask");
    std::vector<std::vector<float>> logits;
    if(run_root_duration_probe_batch(*runtime,global,gm,local,lm,poses,pm,batch,logits,reason)!=MB_OK)throw std::runtime_error(reason);
    if(logits.size()!=batch)throw std::runtime_error("duration batch count mismatch");
    for(const auto & v:logits)if(v.size()!=12||!std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("invalid duration output");
    std::cout<<"duration capacity passed batch="<<batch<<std::endl;
    std::vector<root_result> roots;
    if(run_root_planner_auto_probe_batch(*runtime,global,gm,local,lm,poses,pm,6,batch,roots,reason)!=MB_OK)throw std::runtime_error(reason);
    if(roots.size()!=batch)throw std::runtime_error("root batch count mismatch");
    for(const auto & r:roots)if(r.global_root_values.size()!=24*5||!std::all_of(r.global_root_values.begin(),r.global_root_values.end(),[](float x){return std::isfinite(x);}))
        throw std::runtime_error("invalid root output");
    std::cout<<"root capacity passed batch="<<batch<<" precision="<<precision<<'\n';return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
