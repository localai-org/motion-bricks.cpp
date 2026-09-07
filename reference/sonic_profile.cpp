// Linux diagnostic interposer. No production runtime or arithmetic changes.
// Enable only around the single-caller reference/run_native_sonic.py capture.
#include <motionbricks/physics.h>
#include <ggml-backend.h>
#include <mujoco/mujoco.h>
#include <dlfcn.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
using Clock=std::chrono::steady_clock;
struct Metric {const char * name;uint64_t count=0,ns=0,bytes=0;};
std::array<Metric,7> metrics{{{"tick"},{"encode"},{"decode"},{"mj_step"},{"graph"},{"upload"},{"download"}}};
thread_local bool active=false;
struct Timer {size_t i;bool enabled;Clock::time_point start;
    Timer(size_t index,uint64_t bytes=0):i(index),enabled(active),start(Clock::now()) {if(enabled)metrics[i].bytes+=bytes;}
    ~Timer(){if(enabled){++metrics[i].count;metrics[i].ns+=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());}}
};
template<class T>T next(const char * name) {auto p=dlsym(RTLD_NEXT,name);if(!p){std::fprintf(stderr,"missing %s: %s\n",name,dlerror());std::abort();}return reinterpret_cast<T>(p);}
__attribute__((destructor)) void report(){
    const char * path=std::getenv("MOTIONBRICKS_PROFILE_OUTPUT");
    FILE * out=path?std::fopen(path,"wx"):stderr;
    if(!out){std::perror("profile output");return;}
    std::fprintf(out,"[");bool first=true;
    for(const auto & m:metrics){std::fprintf(out,"%s{\"name\":\"%s\",\"calls\":%llu,\"total_ms\":%.6f,\"mean_ms\":%.6f,\"bytes\":%llu}",first?"":",",m.name,(unsigned long long)m.count,double(m.ns)/1e6,m.count?double(m.ns)/1e6/double(m.count):0.,(unsigned long long)m.bytes);first=false;}
    std::fprintf(out,"]\n");if(path)std::fclose(out);
}
}
extern "C" {
mb_status mb_physics_step(mb_physics * s,const float * roots,uint64_t nr,const float * rot,uint64_t nq,uint32_t frames,double t,float * actual,uint64_t na,float * reference,uint64_t nb,char * error,uint64_t cap){
    static auto f=next<decltype(&mb_physics_step)>("mb_physics_step");active=true;mb_status result;
    {Timer timer(0);result=f(s,roots,nr,rot,nq,frames,t,actual,na,reference,nb,error,cap);}active=false;return result;
}
mb_status mb_sonic_encode(mb_sonic * m,const float * a,uint64_t n,float * b,uint64_t nb,char * error,uint64_t cap){static auto f=next<decltype(&mb_sonic_encode)>("mb_sonic_encode");Timer timer(1);return f(m,a,n,b,nb,error,cap);}
mb_status mb_sonic_decode(mb_sonic * m,const float * a,uint64_t n,float * b,uint64_t nb,char * error,uint64_t cap){static auto f=next<decltype(&mb_sonic_decode)>("mb_sonic_decode");Timer timer(2);return f(m,a,n,b,nb,error,cap);}
void mj_step(const mjModel * m,mjData * d){static auto f=next<decltype(&mj_step)>("mj_step");Timer timer(3);f(m,d);}
ggml_status ggml_backend_graph_compute(ggml_backend_t b,ggml_cgraph * g){static auto f=next<decltype(&ggml_backend_graph_compute)>("ggml_backend_graph_compute");Timer timer(4);return f(b,g);}
void ggml_backend_tensor_set(ggml_tensor * t,const void * d,size_t offset,size_t size){static auto f=next<decltype(&ggml_backend_tensor_set)>("ggml_backend_tensor_set");Timer timer(5,size);f(t,d,offset,size);}
void ggml_backend_tensor_get(const ggml_tensor * t,void * d,size_t offset,size_t size){static auto f=next<decltype(&ggml_backend_tensor_get)>("ggml_backend_tensor_get");Timer timer(6,size);f(t,d,offset,size);}
}
