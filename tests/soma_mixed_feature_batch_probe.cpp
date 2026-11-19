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
#include <fstream>
#include "neural_precision.hpp"
#include "soma_trial.hpp"
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
    if(argc<6)throw std::runtime_error("usage: soma-mixed-feature-batch-probe BUNDLE cpu|vulkan REPORT.json FIXTURE...");
    const std::string backend=argv[2];if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    configure_soma_trial(backend,true);
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    std::vector<soma_request> requests;std::vector<soma_result> singles;
    std::vector<soma_boundaries> single_boundaries;
    std::vector<std::vector<float>> references,reference_logits;
    std::vector<unsigned> durations;std::vector<std::string> hashes;
    for(int arg=4;arg<argc;++arg){
        fixture f(argv[arg]);soma_request request;request.mixed_fp16=true;
        const auto metadata=[&](const char * key){const auto index=gguf_find_key(f.metadata,key);
            if(index<0)throw std::runtime_error(key);return std::string(gguf_get_val_str(f.metadata,index));};
        if(metadata("fixture.precision")!="mixed-fp16")throw std::runtime_error("expected mixed-FP16 fixture");
        if(metadata("motionbricks.upstream_revision")!="54fc563aeaaf819a753d72cc9fdc6713989cdd43")
            throw std::runtime_error("reference revision mismatch");
        hashes.push_back(metadata("motionbricks.source_sha256"));
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

        soma_result single;soma_boundaries boundaries;
        if(run_soma_features(*runtime,request,single,reason,&boundaries)!=MB_OK)throw std::runtime_error(reason);
        single_boundaries.push_back(std::move(boundaries));
        singles.push_back(std::move(single));requests.push_back(request);
        durations.push_back(static_cast<unsigned>(f.read<int>("input.num_tokens").at(0)));
        references.push_back(f.read<float>("full.output"));reference_logits.push_back(f.read<float>("output.pose_logits"));
    }
    std::ofstream report(argv[3]);
    report<<"{\"format\":\"motionbricks-soma-mixed-feature-batch-diagnostic-v1\",\"input_scope\":\"captured raw features; no native skeleton encoding\",\"backend\":\""<<backend
          <<"\",\"status\":\"diagnostic; parity not established\",\"matmul_operands\":\""
          <<(native_half_operands_requested()?"f16":"f32-rounded-to-f16")<<"\",\"aligned_matmul_trial\":"<<(neural_aligned_matmul_trial?"true":"false")
          <<",\"welford_trial\":"<<(neural_welford_trial?"true":"false")
          <<",\"softmax_trial\":"<<(neural_softmax_trial?"true":"false")
          <<",\"rsqrt_table_sha256\":\""<<(soma_rsqrt_corrections.empty()?"":soma_rsqrt_table_sha256)
          <<"\",\"exp2_table_sha256\":\""<<(neural_softmax_trial?soma_exp2_table_sha256:"")
          <<"\",\"rows\":[";
    bool comma=false;
    for(unsigned order=0;order<2;++order){
        std::vector<soma_result> batch;std::vector<soma_boundaries> batch_boundaries;
        if(run_soma_features_batch(*runtime,requests,batch,reason,&batch_boundaries)!=MB_OK)throw std::runtime_error(reason);
        if(batch.size()!=requests.size())throw std::runtime_error("batch result count mismatch");
        for(std::size_t i=0;i<batch.size();++i){
            const auto & b=batch[i];const auto & s=singles[i];const auto duration=durations[i];
            const auto compared=std::min(b.pose_tokens.size(),std::size_t(duration*5));
            if(reference_logits[i].size()<duration*80||references[i].size()!=64*366)throw std::runtime_error("invalid reference shape");
            std::size_t mismatches=0,single_mismatches=0,single_reference_mismatches=0;
            for(std::size_t j=0;j<compared;++j){const auto first=reference_logits[i].begin()+j*16;
                mismatches+=b.pose_tokens[j]!=std::max_element(first,first+16)-first;}
            for(std::size_t j=0;j<std::min(b.pose_tokens.size(),s.pose_tokens.size());++j)single_mismatches+=b.pose_tokens[j]!=s.pose_tokens[j];
            for(std::size_t j=0;j<std::min(s.pose_tokens.size(),std::size_t(duration*5));++j){
                const auto first=reference_logits[i].begin()+j*16;
                single_reference_mismatches+=s.pose_tokens[j]!=std::max_element(first,first+16)-first;
            }
            double maximum=0;
            for(std::size_t j=0;j<std::min(b.features.size(),std::size_t(duration*4*366));++j){
                if(!std::isfinite(b.features[j])||!std::isfinite(references[i][j]))throw std::runtime_error("nonfinite output");
                maximum=std::max(maximum,std::abs(double(b.features[j])-references[i][j]));
            }
            if(comma)report<<",";comma=true;
            report<<"{\"order\":"<<order<<",\"index\":"<<i<<",\"capture_sha256\":\""<<hashes[i]
                  <<"\",\"reference_tokens\":"<<duration<<",\"native_tokens\":"<<b.tokens
                  <<",\"pose_token_mismatches\":"<<mismatches<<",\"compared_pose_tokens\":"<<compared
                  <<",\"raw_feature_common_prefix_max\":"<<maximum<<",\"single_tokens\":"<<s.tokens
                  <<",\"single_reference_pose_mismatches\":"<<single_reference_mismatches
                  <<",\"batch_single_pose_mismatches\":"<<single_mismatches<<",\"batch_single_feature_max\":";
            if(b.features.size()==s.features.size())report<<error(b.features,s.features);else report<<"null";
            report<<",\"batch_single_boundaries\":{";bool boundary_comma=false;
            for(const auto & [name,values]:batch_boundaries[i]){
                const auto & expected=single_boundaries[i].at(name);
                if(boundary_comma)report<<",";boundary_comma=true;
                report<<"\""<<name<<"\":{\"shape_matches\":"<<(values.size()==expected.size()?"true":"false");
                if(values.size()==expected.size()){
                    double difference=0;std::size_t unequal=0;
                    for(std::size_t j=0;j<values.size();++j){
                        if(std::isnan(values[j])&&std::isnan(expected[j]))continue;
                        if(!std::isfinite(values[j])||!std::isfinite(expected[j]))throw std::runtime_error("nonfinite boundary mismatch");
                        difference=std::max(difference,std::abs(double(values[j])-expected[j]));unequal+=values[j]!=expected[j];
                    }
                    report<<",\"max_absolute\":"<<difference<<",\"unequal_values\":"<<unequal;
                }
                report<<"}";
            }
            report<<"}}";
            std::cout<<"order="<<order<<" index="<<i<<" ref_tokens="<<duration<<" native_tokens="<<b.tokens
                     <<" pose_mismatches="<<mismatches<<" raw_feature_max="<<maximum<<" batch_single_pose_mismatches="<<single_mismatches<<'\n';
        }
        std::reverse(requests.begin(),requests.end());std::reverse(singles.begin(),singles.end());
        std::reverse(single_boundaries.begin(),single_boundaries.end());
        std::reverse(references.begin(),references.end());std::reverse(reference_logits.begin(),reference_logits.end());
        std::reverse(durations.begin(),durations.end());std::reverse(hashes.begin(),hashes.end());
    }
    report<<"]}\n";report.close();if(!report)throw std::runtime_error("cannot write report");
    return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
