#include <motionbricks/inference.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t * b,size_t size) {
 if(size<8)return 0;
 char error[128];const uint64_t ec=b[0]%129;
 mb_inference_request * r=nullptr;mb_runtime_options * o=nullptr;
 if(mb_inference_request_create(&r,error,ec)!=MB_OK)return 0;
 mb_runtime_options_create(&o,error,ec);
 mb_runtime_options_set_model_kind(o,b[1],error,ec);uint32_t value;
 mb_runtime_options_get_model_kind(o,&value,error,ec);
 mb_runtime_options_set_device(o,b[2],error,ec);
 mb_inference_request_set_precision(r,b[3],error,ec);
 mb_inference_request_get_precision(r,&value,error,ec);
 float heading=0;std::memcpy(&heading,b+4,4);
 mb_inference_request_set_heading(r,heading,error,ec);
 mb_inference_request_get_heading(r,&heading,error,ec);
 std::array<int32_t,256> parents{};std::array<uint32_t,30> mapping{};
 std::array<float,256*7> reference{};std::array<float,4*256*7> poses{};
 for(unsigned i=0;i<256;++i){parents[i]=i?int(i)-1:-1;reference[i*7+6]=1;}
 for(unsigned i=0;i<30;++i)mapping[i]=i;
 auto status=mb_inference_request_set_skeleton(r,30,parents.data(),reference.data(),210,mapping.data(),30,1,2,error,ec);
 assert(status==MB_OK);
 for(unsigned f=0;f<4;++f)std::copy_n(reference.data(),210,poses.data()+f*210);
 assert(mb_inference_request_set_boundary_transforms(r,0,poses.data(),840,error,ec)==MB_OK);
 if(size>8)std::memcpy(poses.data(),b+8,std::min(size-8,sizeof(poses)));
 const uint64_t count=b[4]&1?840:b[5];
 status=mb_inference_request_set_boundary_transforms(r,b[6]%4,b[7]&1?poses.data():nullptr,count,error,ec);
 // Rejected setters must leave a valid, unchanged source boundary.
 std::array<float,840> read{};uint64_t n=0;
 assert(mb_inference_request_get_boundary_transforms(r,0,read.data(),read.size(),&n,error,ec)==MB_OK&&n==840);
 if(status!=MB_OK)for(unsigned i=0;i<840;++i)assert(read[i]==(i%7==6?1.F:0.F));
 parents[b[1]]=int32_t(b[2])-2;mapping[b[3]%30]=b[4];reference[b[5]]=heading;
 mb_inference_request_set_skeleton(r,b[6]&1?30:256,parents.data(),reference.data(),b[6]&1?210:1792,mapping.data(),b[7]%32,b[1],b[2],error,ec);
 uint32_t mask[11];for(unsigned i=0;i<11;++i)mask[i]=b[i%size]%3;
 mb_inference_request_set_mask(r,3,mask,11,error,ec);
 mb_inference_request_get_boundary_transforms(r,b[2]%4,poses.data(),b[3],&n,error,ec);
 static mb_model * model=[](){
     const char * path=std::getenv("MOTIONBRICKS_FUZZ_SOMA_BUNDLE");if(!path)return static_cast<mb_model *>(nullptr);
     char e[512];mb_runtime_options * options=nullptr;mb_model * m=nullptr;
     mb_runtime_options_create(&options,e,sizeof e);
     mb_runtime_options_set_model_kind(options,MB_MODEL_SOMA,e,sizeof e);
     mb_runtime_options_set_device(options,MB_DEVICE_CPU,e,sizeof e);
     mb_runtime_options_set_threads(options,1,e,sizeof e);
     auto status=mb_model_load(path,options,&m,e,sizeof e);mb_runtime_options_free(options);
     if(status!=MB_OK)std::abort();return m;
 }();
 mb_motion * motion=nullptr;
 if(model&&(b[0]&63)==0) {
     // Exercise model-specific rejection and execution with a trusted model.
     mb_inference_request_set_boundary_transforms(r,1,reference.data(),210,error,ec);
     mb_model_infer(model,r,&motion,error,ec);mb_motion_free(motion);
 } else {mb_model_infer(nullptr,r,&motion,error,ec);assert(!motion);}
 mb_inference_request_free(r);mb_runtime_options_free(o);return 0;
}
