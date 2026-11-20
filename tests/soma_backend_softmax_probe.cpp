// Isolate backend softmax on independently captured CUDA logits.
#include "neural_runtime.hpp"
#include <ggml.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>
using namespace motionbricks::detail;
int main(int argc,char ** argv) try {
    if(argc!=5)throw std::runtime_error("usage: soma-backend-softmax-probe BUNDLE FIXTURE OUTPUT.json cpu|vulkan");
    const std::string backend=argv[4];
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    std::ifstream input(argv[2],std::ios::binary);
    auto read=[&](auto & value){input.read(reinterpret_cast<char *>(&value),sizeof(value));};
    uint32_t count=0;read(count);
    if(!input||count!=24)throw std::runtime_error("invalid softmax fixture layer count");
    std::ofstream report(argv[3]);
    std::ofstream tensors(std::string(argv[3])+".bin",std::ios::binary);
    tensors.write(reinterpret_cast<const char *>(&count),sizeof(count));
    report<<"{\"format\":\"motionbricks-soma-backend-softmax-diagnostic-v1\",\"backend\":\""<<backend<<"\",\"layers\":[";
    size_t total=0,float_errors=0,half_errors=0;double maximum=0;
    for(uint32_t layer=0;layer<count;++layer) {
        uint32_t width=0,rows=0;read(width);read(rows);
        if(!input||!width||width>1024||!rows||rows>4096)throw std::runtime_error("invalid softmax fixture shape");
        std::vector<float> x(size_t(width)*rows),expected(x.size()),actual(x.size());
        input.read(reinterpret_cast<char *>(x.data()),x.size()*sizeof(float));
        input.read(reinterpret_cast<char *>(expected.data()),expected.size()*sizeof(float));
        if(!input)throw std::runtime_error("truncated fixture");
        std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({1024*1024,nullptr,true}),ggml_free);
        if(!context)throw std::runtime_error("cannot allocate graph context");
        auto * source=ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,width,rows);
        auto * output=ggml_soft_max(context.get(),source);
        auto * graph=ggml_new_graph(context.get());ggml_build_forward_expand(graph,output);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(),neural_backend(*runtime)),ggml_backend_buffer_free);
        if(!buffer)throw std::runtime_error("cannot allocate softmax buffer");
        ggml_backend_tensor_set(source,x.data(),0,x.size()*sizeof(float));
        if(neural_graph_compute(*runtime,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("softmax compute failed");
        ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
        size_t differences=0,half=0;double error=0;
        for(size_t i=0;i<x.size();++i) {
            if(!std::isfinite(actual[i])||!std::isfinite(expected[i]))throw std::runtime_error("nonfinite softmax output");
            differences+=actual[i]!=expected[i];half+=ggml_fp32_to_fp16(actual[i])!=ggml_fp32_to_fp16(expected[i]);
            error=std::max(error,std::abs(double(actual[i])-expected[i]));
        }
        if(layer)report<<',';
        report<<"{\"index\":"<<layer<<",\"width\":"<<width<<",\"rows\":"<<rows<<",\"values\":"<<x.size()
              <<",\"float_mismatches\":"<<differences<<",\"half_mismatches\":"<<half<<",\"max_absolute\":"<<error<<'}';
        tensors.write(reinterpret_cast<const char *>(&width),sizeof(width));
        tensors.write(reinterpret_cast<const char *>(&rows),sizeof(rows));
        tensors.write(reinterpret_cast<const char *>(actual.data()),actual.size()*sizeof(float));
        total+=x.size();float_errors+=differences;half_errors+=half;maximum=std::max(maximum,error);
    }
    if(input.peek()!=std::char_traits<char>::eof()||total!=93312)throw std::runtime_error("unexpected fixture inventory");
    report<<"],\"values\":"<<total<<",\"float_mismatches\":"<<float_errors<<",\"half_mismatches\":"<<half_errors
          <<",\"max_absolute\":"<<maximum<<",\"status\":\"isolated diagnostic; not full model parity\"}\n";
    report.close();tensors.close();if(!report||!tensors)throw std::runtime_error("cannot write diagnostic output");
    std::cout<<"values="<<total<<" float_mismatches="<<float_errors<<" half_mismatches="<<half_errors<<" max="<<maximum<<'\n';
    return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
