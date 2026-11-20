#include "neural_runtime.hpp"
#include "pose.hpp"
#include "soma.hpp"
#include "root.hpp"
#include "decoder.hpp"
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
double error(const std::vector<float> & a,const std::vector<float> & b) {
    if(a.size()!=b.size())throw std::runtime_error("output shape mismatch");
    double e=0;
    for(std::size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite batch output");
        e=std::max(e,std::abs(double(a[i])-b[i]));
    }
    return e;
}
int main(int argc,char ** argv) try {
    if(argc<5)throw std::runtime_error("usage: soma-feature-batch-parity BUNDLE cpu|vulkan FIXTURE...");
    const std::string backend=argv[2];
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="vulkan"?MB_DEVICE_VULKAN:MB_DEVICE_CPU,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    std::vector<soma_request> requests;std::vector<soma_result> singles;std::vector<std::vector<float>> references;
    for(int arg=3;arg<argc;++arg){
        fixture f(argv[arg]);soma_request request;
        const auto floats=[&](const char * key,auto & target){auto value=f.read<float>(key);
            if(value.size()!=target.size())throw std::runtime_error("input shape mismatch");
            std::copy(value.begin(),value.end(),target.begin());};
        const auto masks=[&](const char * key,auto & target){auto value=f.read<std::int32_t>(key);
            if(value.size()!=target.size())throw std::runtime_error("mask shape mismatch");
            std::copy(value.begin(),value.end(),target.begin());};
        floats("full.global_root_values",request.global_root);floats("full.local_root_values",request.local_root);
        floats("full.local_poses",request.poses);masks("full.has_global_root_values",request.has_global);
        masks("full.has_local_root_values",request.has_local);masks("full.has_local_poses",request.has_pose);
        masks("full.allowed_pred_num_tokens",request.allowed);
        soma_result single;
        if(run_soma_features(*runtime,request,single,reason)!=MB_OK)throw std::runtime_error(reason);
        if(single.tokens!=static_cast<unsigned>(f.read<int>("input.num_tokens")[0]))throw std::runtime_error("single duration differs from reference");
        const auto reference_logits=f.read<float>("output.pose_logits");
        if(reference_logits.size()<single.pose_tokens.size()*16)throw std::runtime_error("reference logits are incomplete");
        for(std::size_t i=0;i<single.pose_tokens.size();++i){
            const auto first=reference_logits.begin()+i*16;
            if(single.pose_tokens[i]!=std::max_element(first,first+16)-first)
                throw std::runtime_error("single pose token differs from reference");
        }
        auto expected=f.read<float>("full.output");expected.resize(single.features.size());
        references.push_back(std::move(expected));requests.push_back(request);singles.push_back(std::move(single));
    }
    double reference_max=0,single_max=0;
    for(unsigned order=0;order<2;++order){
        std::vector<soma_result> batch;std::vector<soma_boundaries> taps;
        if(run_soma_features_batch(*runtime,requests,batch,reason,&taps)!=MB_OK)throw std::runtime_error(reason);
        if(batch.size()!=requests.size()||taps.size()!=requests.size())throw std::runtime_error("batch output count mismatch");
        for(std::size_t i=0;i<batch.size();++i){
            if(batch[i].tokens!=singles[i].tokens||batch[i].pose_tokens!=singles[i].pose_tokens||
               batch[i].used_root_predictor!=singles[i].used_root_predictor)throw std::runtime_error("batch decision differs from single");
            if(taps[i].at("output_features")!=batch[i].features)throw std::runtime_error("batch boundary order mismatch");
            reference_max=std::max(reference_max,error(batch[i].features,references[i]));
            single_max=std::max(single_max,error(batch[i].features,singles[i].features));
        }
        std::reverse(requests.begin(),requests.end());std::reverse(singles.begin(),singles.end());std::reverse(references.begin(),references.end());
    }
    // Precision isolation is a separate contract, not CUDA mixed-FP16 parity.
    // Compare each precision group with that same group executed independently;
    // keep batch shapes identical so ordinary batching arithmetic is not hidden.
    auto mixed=requests;
    for(std::size_t i=0;i+1<mixed.size();++i)mixed[i].mixed_fp16=true;
    std::vector<soma_result> half_group,float_group,combined;
    const auto all=std::span<const soma_request>(mixed);
    if(run_soma_features_batch(*runtime,all.first(mixed.size()-1),half_group,reason)!=MB_OK||
       run_soma_features_batch(*runtime,all.last(1),float_group,reason)!=MB_OK||
       run_soma_features_batch(*runtime,all,combined,reason)!=MB_OK)throw std::runtime_error(reason);
    if(combined.size()!=mixed.size()||half_group.size()!=mixed.size()-1||float_group.size()!=1)
        throw std::runtime_error("precision group result count mismatch");
    for(std::size_t i=0;i<mixed.size();++i){
        const auto & isolated=i+1<mixed.size()?half_group[i]:float_group[0];
        if(combined[i].tokens!=isolated.tokens||combined[i].pose_tokens!=isolated.pose_tokens||
           combined[i].used_root_predictor!=isolated.used_root_predictor||combined[i].features!=isolated.features)
            throw std::runtime_error("mixed precision groups interfere");
    }
    soma_result restored;
    if(run_soma_features(*runtime,requests.back(),restored,reason)!=MB_OK)throw std::runtime_error(reason);
    if(restored.features!=singles.back().features||restored.pose_tokens!=singles.back().pose_tokens)
        throw std::runtime_error("mixed batch changed subsequent FP32 inference");
    std::vector<soma_result> invalid(1);std::vector<soma_boundaries> invalid_taps(1);
    requests.back().allowed.fill(0);
    if(run_soma_features_batch(*runtime,requests,invalid,reason,&invalid_taps)!=MB_INVALID_ARGUMENT||!invalid.empty()||!invalid_taps.empty())
        throw std::runtime_error("invalid batch publishes partial outputs");
    if(run_soma_features_batch(*runtime,{},invalid,reason)!=MB_INVALID_ARGUMENT)throw std::runtime_error("empty batch accepted");
    std::cout<<"requests="<<requests.size()<<" reference_max="<<reference_max<<" single_max="<<single_max
             <<" precision_groups_isolated=true fp32_restored=true\n";
    return reference_max<1e-3&&single_max<1e-3?0:1;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
