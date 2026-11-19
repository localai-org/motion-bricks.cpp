#include <motionbricks/inference.h>
#include "handles.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
int main() {
 char e[512];mb_runtime_options * o=nullptr;assert(mb_runtime_options_create(&o,e,sizeof e)==MB_OK);
 uint32_t v=99;assert(mb_runtime_options_get_model_kind(o,&v,e,sizeof e)==MB_OK&&v==MB_MODEL_G1);
 assert(mb_runtime_options_set_model_kind(o,MB_MODEL_SOMA,e,sizeof e)==MB_OK);
 assert(mb_runtime_options_set_model_kind(o,99,e,sizeof e)==MB_INVALID_ARGUMENT&&strstr(e,"model kind"));
 assert(mb_runtime_options_get_model_kind(o,&v,e,sizeof e)==MB_OK&&v==MB_MODEL_SOMA);
 mb_runtime_options_free(o);
 mb_inference_request * r=nullptr;assert(mb_inference_request_create(&r,e,sizeof e)==MB_OK);
 std::array<int32_t,30> parents{};std::array<uint32_t,30> mapping{};std::array<float,210> ref{};std::array<float,840> context{};
 for(unsigned j=0;j<30;++j){parents[j]=j?int(j)-1:-1;mapping[j]=j;ref[j*7+6]=1;}
 for(unsigned f=0;f<4;++f)std::copy(ref.begin(),ref.end(),context.begin()+f*210);
 auto skeleton=[&](){return mb_inference_request_set_skeleton(r,30,parents.data(),ref.data(),ref.size(),mapping.data(),mapping.size(),1,2,e,sizeof e);};
 assert(skeleton()==MB_OK);
 assert(mb_inference_request_set_boundary_transforms(r,0,context.data(),context.size(),e,sizeof e)==MB_OK);
 assert(mb_inference_request_set_boundary_transforms(r,1,ref.data(),ref.size(),e,sizeof e)==MB_OK);
 mapping[1]=0;assert(skeleton()==MB_INVALID_ARGUMENT&&strstr(e,"duplicate"));mapping[1]=1;
 context[6]=NAN;assert(mb_inference_request_set_boundary_transforms(r,0,context.data(),context.size(),e,sizeof e)==MB_INVALID_ARGUMENT&&e[0]);
 uint64_t count=0;std::array<float,840> copy{};
 assert(mb_inference_request_get_boundary_transforms(r,0,copy.data(),copy.size(),&count,e,sizeof e)==MB_OK&&count==840&&copy[6]==1);
 float sentinel=123;assert(mb_inference_request_get_boundary_transforms(r,0,&sentinel,1,&count,e,sizeof e)==MB_INVALID_ARGUMENT&&sentinel==123);
 assert(mb_inference_request_set_precision(r,3,e,sizeof e)==MB_OK);
 assert(mb_inference_request_set_precision(r,4,e,sizeof e)==MB_INVALID_ARGUMENT&&e[0]);
 assert(mb_inference_request_get_precision(r,&v,e,sizeof e)==MB_OK&&v==3);
 assert(mb_inference_request_set_heading(r,1.5,e,sizeof e)==MB_OK);
 assert(mb_inference_request_set_heading(r,INFINITY,e,sizeof e)==MB_INVALID_ARGUMENT);
 float heading=0;assert(mb_inference_request_get_heading(r,&heading,e,sizeof e)==MB_OK&&heading==1.5);
 mb_model g1,soma;soma.kind=MB_MODEL_SOMA;mb_motion * out=reinterpret_cast<mb_motion *>(1);
 assert(mb_model_infer(&g1,r,&out,e,sizeof e)==MB_INCOMPATIBLE_MODEL&&!out&&strstr(e,"G1"));
 mb_agent * agent=reinterpret_cast<mb_agent *>(1);
 assert(mb_agent_create(&soma,&agent,e,sizeof e)==MB_INCOMPATIBLE_MODEL&&!agent&&strstr(e,"mb_model_infer"));
 assert(mb_inference_request_set_seed(r,0,e,sizeof e)==MB_OK);
 assert(mb_model_infer(&soma,r,&out,e,sizeof e)==MB_INVALID_ARGUMENT&&!out&&strstr(e,"sampling"));
 // Reconfiguration clears boundaries; failed reconfiguration above did not.
 assert(skeleton()==MB_OK);assert(mb_inference_request_get_boundary_transforms(r,0,nullptr,0,&count,e,sizeof e)==MB_OK&&count==0);
 mb_inference_request_free(r);
}
