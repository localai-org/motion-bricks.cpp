#include <motionbricks/inference.h>
#include "agent.hpp"
#include "handles.hpp"
#include "planner.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
extern "C" mb_status infer_between(const mb_model *,const float *,const float *,uint64_t,
                                  mb_motion **,char *,uint64_t);
#define CHECK(x) do {if(!(x)){std::cerr<<"line "<<__LINE__<<": "<<#x<<": "<<error<<'\n';return 1;}}while(0)
int main(int argc,char ** argv) {
    if(argc<3)return 1;
    char error[1024]{};mb_runtime_options * options=nullptr;mb_model * model=nullptr;
    CHECK(mb_runtime_options_create(&options,error,sizeof error)==MB_OK);
    CHECK(mb_runtime_options_set_device(options,argc>3?MB_DEVICE_VULKAN:MB_DEVICE_CPU,error,sizeof error)==MB_OK);
    CHECK(mb_runtime_options_set_threads(options,4,error,sizeof error)==MB_OK);
    CHECK(mb_model_load(argv[1],options,&model,error,sizeof error)==MB_OK);
    mb_runtime_options_free(options);
    mb_inference_request * request=nullptr;
    CHECK(mb_inference_request_create(&request,error,sizeof error)==MB_OK);
    mb_motion * output=nullptr;
    CHECK(mb_model_infer(model,request,&output,error,sizeof error)==MB_INVALID_ARGUMENT&&!output);
    // Obtain realistic sparse constraints from the existing traced controller.
    // Only the input capture uses a style; the public inference calls below do not.
    mb_style * style=nullptr;mb_agent * agent=nullptr;
    CHECK(mb_style_load(model,argv[2],&style,error,sizeof error)==MB_OK);
    CHECK(mb_agent_create(model,&agent,error,sizeof error)==MB_OK);
    CHECK(mb_agent_reset(agent,style,error,sizeof error)==MB_OK);
    mb_command command;command.style=style;command.movement_direction={0,0,1};command.seed=23;
    motionbricks::detail::transition_trace trace;mb_motion planned;std::string reason;
    CHECK(motionbricks::detail::plan_agent_trace(*agent,command,planned,trace,reason)==MB_OK);
    mb_agent_free(agent);mb_style_free(style);
    const auto original=trace.constraints;
    CHECK(mb_inference_request_set_features(request,0,original.global_root.data(),40,error,sizeof error)==MB_OK);
    CHECK(mb_inference_request_set_features(request,1,original.local_root.data(),32,error,sizeof error)==MB_OK);
    CHECK(mb_inference_request_set_features(request,2,original.poses.data(),2424,error,sizeof error)==MB_OK);
    // Both samplers, shortest and longest duration, and a sparse target mask.
    for(uint32_t argmax=0;argmax<2;++argmax)for(uint32_t duration:{0U,10U}) {
        auto constraints=original;constraints.allowed_tokens.fill(0);constraints.allowed_tokens[duration]=1;
        constraints.has_poses[5]=0;
        std::array<uint32_t,11> durations{};durations[duration]=1;
        std::array<uint32_t,8> poses{};std::copy(constraints.has_poses.begin(),constraints.has_poses.end(),poses.begin());
        CHECK(mb_inference_request_set_mask(request,3,durations.data(),11,error,sizeof error)==MB_OK);
        CHECK(mb_inference_request_set_mask(request,2,poses.data(),8,error,sizeof error)==MB_OK);
        CHECK(mb_inference_request_set_seed(request,23,error,sizeof error)==MB_OK);
        CHECK(mb_inference_request_set_sampling_argmax(request,argmax,error,sizeof error)==MB_OK);
        mb_motion expected;
        CHECK(motionbricks::detail::run_transition(*model,constraints,expected,nullptr,nullptr,reason,23,argmax!=0)==MB_OK);
        CHECK(mb_model_infer(model,request,&output,error,sizeof error)==MB_OK);
        CHECK(output->frames==(6+duration)*4&&output->joints==34&&output->target_frames==0);
        CHECK(output->root_translations==expected.root_translations&&output->local_rotations_xyzw==expected.local_rotations_xyzw);
        mb_motion * repeated=nullptr;
        CHECK(mb_model_infer(model,request,&repeated,error,sizeof error)==MB_OK);
        CHECK(output->root_translations==repeated->root_translations&&output->local_rotations_xyzw==repeated->local_rotations_xyzw);
        mb_motion_free(repeated);mb_motion_free(output);output=nullptr;
    }
    // Pose convenience adapter matches the internal encoder exactly; target's
    // final velocity is repeated. Includes moving, turned source poses.
    std::array<float,12> roots{};std::array<float,544> rotations{};
    for(size_t f=0;f<4;++f) {
        roots[f*3]=float(f)*.01F;roots[f*3+1]=.8F;
        for(size_t j=0;j<34;++j)rotations[(f*34+j)*4+3]=1;
        rotations[f*136+1]=std::sin(.1F*float(f));rotations[f*136+3]=std::cos(.1F*float(f));
    }
    motionbricks::detail::encoded_frames encoded;
    CHECK(motionbricks::detail::encode_context(*model,roots,rotations,4,encoded,reason)==MB_OK);
    for(uint32_t b=0;b<2;++b)CHECK(mb_inference_request_set_boundary_poses(request,model,b,roots.data(),12,rotations.data(),544,error,sizeof error)==MB_OK);
    std::array<float,40> global{};uint64_t count=0;
    CHECK(mb_inference_request_get_features(request,0,global.data(),40,&count,error,sizeof error)==MB_OK);
    CHECK(std::equal(encoded.global_root.begin(),encoded.global_root.end(),global.begin()));
    CHECK(std::equal(encoded.global_root.begin(),encoded.global_root.end(),global.begin()+20));
    std::array<float,32> local{};
    CHECK(mb_inference_request_get_features(request,1,local.data(),32,&count,error,sizeof error)==MB_OK);
    CHECK(std::equal(encoded.local_root.begin(),encoded.local_root.end(),local.begin()));
    CHECK(local[28]==local[24]&&local[29]==local[25]&&local[30]==local[26]&&local[31]==global[36]);
    std::array<float,2424> pose{};
    CHECK(mb_inference_request_get_features(request,2,pose.data(),2424,&count,error,sizeof error)==MB_OK);
    CHECK(std::equal(encoded.poses.begin(),encoded.poses.end(),pose.begin()));
    CHECK(std::equal(encoded.poses.begin(),encoded.poses.end(),pose.begin()+1212));
    rotations[3]=0;rotations[1]=0;
    CHECK(mb_inference_request_set_boundary_poses(request,model,0,roots.data(),12,rotations.data(),544,error,sizeof error)==MB_INVALID_ARGUMENT);
    std::array<float,40> unchanged{};
    CHECK(mb_inference_request_get_features(request,0,unchanged.data(),40,&count,error,sizeof error)==MB_OK&&unchanged==global);
    CHECK(mb_model_infer(model,request,&output,error,sizeof error)==MB_OK);
    std::array<float,24> example_roots{};std::array<float,1088> example_rotations{};
    rotations[3]=1;
    for(size_t b=0;b<2;++b) {
        std::copy(roots.begin(),roots.end(),example_roots.begin()+b*12);
        std::copy(rotations.begin(),rotations.end(),example_rotations.begin()+b*544);
    }
    mb_motion * example_output=nullptr;
    CHECK(infer_between(model,example_roots.data(),example_rotations.data(),23,&example_output,error,sizeof error)==MB_OK);
    CHECK(example_output->frames==40&&example_output->joints==34);
    mb_motion_free(example_output);
    mb_inference_request_free(request);mb_model_free(model);
    // Results remain valid after both inputs have been destroyed.
    const float * borrowed=nullptr;
    CHECK(mb_motion_get_root_translations(output,&borrowed,&count,error,sizeof error)==MB_OK&&count>0&&std::isfinite(borrowed[0]));
    mb_motion_free(output);
    std::cout<<"stateless inference: exact internal-boundary equivalence, repeatability, adapters, ownership passed\n";
}
