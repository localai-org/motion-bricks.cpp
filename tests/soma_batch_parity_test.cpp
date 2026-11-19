#include "neural_runtime.hpp"
#include "pose.hpp"
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
    if(argc<4||argc>5)throw std::runtime_error("usage: soma-batch-parity BUNDLE FIXTURE_A FIXTURE_B [cpu|vulkan]");
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    const auto device=argc==5&&std::string(argv[4])=="vulkan"?MB_DEVICE_VULKAN:MB_DEVICE_CPU;
    if(create_neural_runtime(argv[1],device,1,"",runtime,reason)!=MB_OK)throw std::runtime_error(reason);
    fixture a(argv[2]),b(argv[3]);
    const auto duration=a.read<int>("input.num_tokens").at(0);
    if(duration!=16||b.read<int>("input.num_tokens").at(0)!=duration)throw std::runtime_error("expected two 16-token fixtures");
    if(a.read<float>("root.poses")==b.read<float>("root.poses"))throw std::runtime_error("batch requests must differ");
    const auto check=[&](mb_status status){if(status!=MB_OK)throw std::runtime_error(reason);};
    double pose_max=0,root_max=0,duration_max=0,decoder_max=0;
    std::size_t token_mismatches=0,duration_mismatches=0;
    // Reverse the batch as well: output order must follow input order.
    for(unsigned order=0;order<2;++order) {
        const std::vector<const fixture *> items=order?std::vector<const fixture *>{&b,&a}:std::vector<const fixture *>{&a,&b};
        const auto floats=[&](const char * name) {
            std::vector<float> values;
            for(auto * item:items){auto v=item->read<float>(name);values.insert(values.end(),v.begin(),v.end());}
            return values;
        };
        const auto integers=[&](const char * name) {
            std::vector<std::int32_t> values;
            for(auto * item:items){auto v=item->read<std::int32_t>(name);values.insert(values.end(),v.begin(),v.end());}
            return values;
        };
        const auto masks=[&](const char * name) {auto values=integers(name);return std::vector<std::uint8_t>(values.begin(),values.end());};
        std::vector<std::vector<float>> pose,logits,decoded;
        std::vector<root_result> roots;
        check(run_pose_planner_batch(*runtime,integers("input.pose_tokens"),floats("input.root_values"),
            floats("input.pose_cond"),masks("input.has_pose_cond"),16,16,2,pose,reason));
        check(run_root_duration_probe_batch(*runtime,floats("root.global"),masks("root.global_mask"),
            floats("root.local"),masks("root.local_mask"),floats("root.poses"),masks("root.pose_mask"),2,logits,reason));
        check(run_root_planner_auto_probe_batch(*runtime,floats("root.global"),masks("root.global_mask"),
            floats("root.local"),masks("root.local_mask"),floats("root.poses"),masks("root.pose_mask"),16,2,roots,reason));
        check(run_vq_decoder_batch(*runtime,floats("decoder.quantized"),floats("decoder.external"),
            floats("decoder.target"),masks("decoder.mask"),16,2,decoded,reason));
        if(pose.size()!=2||logits.size()!=2||roots.size()!=2||decoded.size()!=2)throw std::runtime_error("batch output count mismatch");
        for(unsigned i=0;i<2;++i) {
            const auto expected=items[i]->read<float>("output.pose_logits");
            pose_max=std::max(pose_max,error(pose[i],expected));
            root_max=std::max(root_max,error(roots[i].global_root_values,items[i]->read<float>("root.motion")));
            duration_max=std::max(duration_max,error(logits[i],items[i]->read<float>("root.logits")));
            const auto expected_duration=items[i]->read<float>("root.logits");
            duration_mismatches+=(std::max_element(logits[i].begin(),logits[i].end())-logits[i].begin())!=
                (std::max_element(expected_duration.begin(),expected_duration.end())-expected_duration.begin());
            duration_max=std::max(duration_max,error(roots[i].duration_logits,items[i]->read<float>("root.logits")));
            decoder_max=std::max(decoder_max,error(decoded[i],items[i]->read<float>("decoder.output")));
            for(unsigned t=0;t<80;++t) {
                const auto begin=t*16;
                token_mismatches+=(std::max_element(pose[i].begin()+begin,pose[i].begin()+begin+16)-(pose[i].begin()+begin))!=
                    (std::max_element(expected.begin()+begin,expected.begin()+begin+16)-(expected.begin()+begin));
            }
        }
    }
    std::cout<<"batch2 pose="<<pose_max<<" root="<<root_max<<" duration="<<duration_max
             <<" decoder="<<decoder_max<<" duration_mismatches="<<duration_mismatches<<" token_mismatches="<<token_mismatches<<'\n';
    return pose_max<1e-3&&root_max<1e-3&&duration_max<1e-3&&decoder_max<1e-3&&token_mismatches==0&&duration_mismatches==0?0:1;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
