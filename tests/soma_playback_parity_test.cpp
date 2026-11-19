#include "soma_playback.hpp"
#include "soma.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace motionbricks::detail;
template<class T> std::vector<T> read(ggml_context * ctx,const char * name) {
    auto * t=ggml_get_tensor(ctx,name);
    if(!t||ggml_nbytes(t)%sizeof(T))throw std::runtime_error(name);
    std::vector<T> result(ggml_nbytes(t)/sizeof(T));
    std::memcpy(result.data(),t->data,ggml_nbytes(t));return result;
}
int main(int argc,char ** argv) try {
    const bool exact_encoding=argc==3 && std::string(argv[2])=="--exact-encoding";
    const bool exact_playback=argc==3 && std::string(argv[2])=="--exact-playback";
    if(argc!=2 && !exact_encoding && !exact_playback)throw std::runtime_error("usage: soma-playback-parity FIXTURE [--exact-encoding|--exact-playback]");
    ggml_context * raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file(gguf_init_from_file(argv[1],{false,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(raw,ggml_free);
    if(!file||!ctx)throw std::runtime_error("invalid playback fixture");
    soma_skeleton skeleton{read<int>(raw,"parents"),read<int>(raw,"pose_to_model"),read<soma_transform>(raw,"reference")};
    soma_playback actual;std::string reason;
    soma_request encoded;
    soma_transform alignment;
    std::vector<soma_transform> world;
    auto target=read<soma_transform>(raw,"target");
    target.resize(read<unsigned>(raw,"target_count").at(0)*skeleton.parents.size());
    const auto hips=read<unsigned>(raw,"heading_joints");
    if(encode_soma_poses(skeleton,read<int>(raw,"model_to_pose"),hips.at(0),hips.at(1),
        read<soma_transform>(raw,"context"),target,{0,0,0,0,0,0,1},2047,encoded,alignment,world,reason)!=MB_OK)
        throw std::runtime_error(reason);
    for(const auto & [name,values]:{std::pair{"global_root_values",std::span<const float>(encoded.global_root)},
        {"local_root_values",std::span<const float>(encoded.local_root)},
        {"local_poses",std::span<const float>(encoded.poses)}}) {
        const auto expected=read<float>(raw,(std::string("input.")+name).c_str());
        if(expected.size()!=values.size())throw std::runtime_error("encoding shape mismatch");
        double maximum=0;
        for(std::size_t i=0;i<values.size();++i) {
            if(!std::isfinite(values[i])||!std::isfinite(expected[i]))throw std::runtime_error("nonfinite encoding");
            maximum=std::max(maximum,std::abs(double(values[i])-expected[i]));
        }
        std::cout<<name<<" max_abs="<<maximum<<'\n';
        if(maximum>((exact_encoding||exact_playback)?0:2e-5))throw std::runtime_error("encoding parity failed");
    }
    for(const auto & [name,values]:{std::pair{"has_global_root_values",std::span<const std::uint8_t>(encoded.has_global)},
        {"has_local_root_values",std::span<const std::uint8_t>(encoded.has_local)},
        {"has_local_poses",std::span<const std::uint8_t>(encoded.has_pose)},
        {"allowed_pred_num_tokens",std::span<const std::uint8_t>(encoded.allowed)}}) {
        const auto expected=read<int>(raw,(std::string("input.")+name).c_str());
        if(!std::equal(values.begin(),values.end(),expected.begin(),expected.end()))throw std::runtime_error("mask mismatch");
    }
    if(exact_encoding)return 0;
    const auto status=decode_soma_playback(read<float>(raw,"features"),read<unsigned>(raw,"tokens").at(0),
        skeleton,read<soma_transform>(raw,"alignment").at(0),{0,0,0,0,0,0,1},
        read<soma_transform>(raw,"context"),read<soma_transform>(raw,"target"),actual,reason);
    if(status!=MB_OK)throw std::runtime_error(reason);
    bool passed=true;
    std::vector<soma_transform> alignments{alignment};
    for(const auto & [name,values]:{std::pair{"decoded",&actual.decoded},{"corrected",&actual.corrected},
        {"world",&actual.world},{"start",&actual.start_deltas},{"end",&actual.end_deltas},
        {"input_world",&world},{"alignment",&alignments}}) {
        const auto expected=read<soma_transform>(raw,name);
        if(values->size()!=expected.size())throw std::runtime_error("shape mismatch");
        double position_error=0,rotation_error=0;
        for(std::size_t i=0;i<expected.size();++i) {
            const auto & a=(*values)[i];const auto & b=expected[i];
            double dot=0;
            for(unsigned j=3;j<7;++j)dot+=double(a[j])*b[j];
            const double sign=dot<0?-1:1;
            for(unsigned j=0;j<7;++j) {
                if(!std::isfinite(a[j]))throw std::runtime_error("nonfinite playback");
                auto & error=j<3?position_error:rotation_error;
                error=std::max(error,std::abs(double(a[j])-(j<3?1:sign)*b[j]));
            }
        }
        std::cout<<name<<" position_max="<<position_error<<" quaternion_max="<<rotation_error<<'\n';
        passed &= exact_playback ? (*values==expected) : (position_error<2e-5 && rotation_error<2e-5);
    }
    const auto contacts=read<float>(raw,"contacts");
    passed &= actual.contacts==contacts;
    return passed?0:1;
} catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
