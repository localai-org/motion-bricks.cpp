#include "soma_playback.hpp"
#include "soma_trial.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace motionbricks::detail;
struct fixture {
    soma_skeleton skeleton;
    std::vector<int> mapping;
    std::vector<soma_transform> context,target,corrected,world;
    std::vector<float> contacts;
    unsigned right{},left{},tokens{};
    explicit fixture(const char * path) {
        ggml_context * raw=nullptr;
        std::unique_ptr<gguf_context,decltype(&gguf_free)> metadata(gguf_init_from_file(path,{false,&raw}),gguf_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors(raw,ggml_free);
        if(!metadata||!raw)throw std::runtime_error("invalid fixture");
        const auto read=[&]<class T>(const char * name,std::vector<T> & out){
            auto * t=ggml_get_tensor(raw,name);
            if(!t||ggml_nbytes(t)%sizeof(T))throw std::runtime_error(name);
            out.resize(ggml_nbytes(t)/sizeof(T));std::memcpy(out.data(),t->data,ggml_nbytes(t));
        };
        read("parents",skeleton.parents);read("pose_to_model",skeleton.pose_to_model);read("reference",skeleton.reference);
        read("model_to_pose",mapping);read("context",context);read("target",target);
        read("corrected",corrected);read("world",world);read("contacts",contacts);
        std::vector<unsigned> count,hips,duration;read("target_count",count);read("heading_joints",hips);read("tokens",duration);
        target.resize(count.at(0)*skeleton.parents.size());right=hips.at(0);left=hips.at(1);tokens=duration.at(0);
    }
    soma_motion_request request() const {
        soma_motion_request r;r.skeleton=&skeleton;r.model_to_pose=mapping;r.context=context;r.target=target;
        r.right_leg=right;r.left_leg=left;return r;
    }
};
struct errors {double position{},rotation{},contact{};};
void compare(std::span<const soma_transform> actual,std::span<const soma_transform> expected,errors & maximum){
    if(actual.size()!=expected.size())throw std::runtime_error("motion dimensions differ");
    for(std::size_t i=0;i<actual.size();++i){
        double square=0,dot=0,anorm=0,bnorm=0;
        for(unsigned j=0;j<7;++j){
            if(!std::isfinite(actual[i][j])||!std::isfinite(expected[i][j]))throw std::runtime_error("nonfinite motion");
            if(j<3){const double d=double(actual[i][j])-expected[i][j];square+=d*d;}
            else {dot+=double(actual[i][j])*expected[i][j];anorm+=double(actual[i][j])*actual[i][j];bnorm+=double(expected[i][j])*expected[i][j];}
        }
        if(anorm<=0||bnorm<=0)throw std::runtime_error("invalid rotation");
        maximum.position=std::max(maximum.position,std::sqrt(square));
        maximum.rotation=std::max(maximum.rotation,2*std::acos(std::clamp(std::abs(dot)/std::sqrt(anorm*bnorm),0.0,1.0))*180/3.141592653589793);
    }
}
int main(int argc,char ** argv) try {
    if(argc<5)throw std::runtime_error("usage: soma-motion-batch-parity BUNDLE cpu|vulkan [mixed-fp16] FIXTURE...");
    const std::string backend=argv[2];if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    const bool mixed=std::string(argv[3])=="mixed-fp16";
    const int first_fixture=mixed?4:3;
    if(argc<first_fixture+2)throw std::runtime_error("batch parity requires at least two fixtures");
    configure_soma_trial(backend,mixed);
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    std::vector<std::unique_ptr<fixture>> fixtures;
    std::vector<soma_motion_request> requests;
    std::vector<soma_motion_result> singles;
    const auto check=[&](mb_status s){if(s!=MB_OK)throw std::runtime_error(reason);};
    for(int i=first_fixture;i<argc;++i){
        fixtures.push_back(std::make_unique<fixture>(argv[i]));auto request=fixtures.back()->request();
        request.mixed_fp16=mixed;
        soma_motion_result s;
        check(encode_soma_poses(*request.skeleton,request.model_to_pose,request.right_leg,request.left_leg,
            request.context,request.target,request.runtime_to_model,request.allowed_token_bits,s.encoded,s.alignment,s.input_world,reason));
        s.encoded.mixed_fp16=mixed;
        check(run_soma_features(*runtime,s.encoded,s.prediction,reason));
        check(decode_soma_playback(s.prediction.features,s.prediction.tokens,*request.skeleton,s.alignment,
            request.model_to_runtime,request.context,request.target,s.playback,reason));
        requests.push_back(request);singles.push_back(std::move(s));
    }
    errors reference,single;
    for(unsigned order=0;order<2;++order){
        std::vector<soma_motion_result> results;
        check(run_soma_motion_batch(*runtime,requests,results,reason,true));
        if(results.size()!=requests.size())throw std::runtime_error("batch result count mismatch");
        for(std::size_t i=0;i<results.size();++i){
            const auto & r=results[i];const auto & f=*fixtures[i];const auto & s=singles[i];
            if(r.prediction.tokens!=f.tokens||r.prediction.tokens!=s.prediction.tokens||r.prediction.pose_tokens!=s.prediction.pose_tokens||
               r.prediction.used_root_predictor!=s.prediction.used_root_predictor)throw std::runtime_error("batch decisions differ");
            if(r.boundaries.at("output_features")!=r.prediction.features||r.input_world!=s.input_world||r.alignment!=s.alignment)
                throw std::runtime_error("processing tap or input order mismatch");
            compare(r.playback.world,f.world,reference);compare(r.playback.corrected,f.corrected,reference);
            compare(r.playback.world,s.playback.world,single);compare(r.playback.corrected,s.playback.corrected,single);
            if(r.playback.contacts.size()!=f.contacts.size())throw std::runtime_error("contacts dimensions differ");
            for(std::size_t j=0;j<f.contacts.size();++j){
                if(!std::isfinite(r.playback.contacts[j]))throw std::runtime_error("nonfinite contacts");
                reference.contact=std::max(reference.contact,std::abs(double(r.playback.contacts[j])-f.contacts[j]));
            }
        }
        std::reverse(fixtures.begin(),fixtures.end());std::reverse(requests.begin(),requests.end());std::reverse(singles.begin(),singles.end());
    }
    std::vector<soma_motion_result> invalid(1);requests.back().skeleton=nullptr;
    if(run_soma_motion_batch(*runtime,requests,invalid,reason)!=MB_INVALID_ARGUMENT||!invalid.empty())
        throw std::runtime_error("invalid batch publishes partial motions");
    std::cout<<"precision="<<(mixed?"mixed-fp16":"fp32")<<" requests="<<requests.size()<<" reference_position_m="<<reference.position<<" reference_rotation_deg="<<reference.rotation
             <<" contacts="<<reference.contact<<" single_position_m="<<single.position<<" single_rotation_deg="<<single.rotation<<'\n';
    return reference.position<=1e-4&&reference.rotation<=0.01&&reference.contact<=1e-4&&single.position<=1e-4&&single.rotation<=0.01?0:1;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
