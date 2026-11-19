#include "soma_softmax.hpp"
#include <iostream>
#include <ggml.h>
int main(int argc,char **argv) try {
 using namespace motionbricks::detail;
 if(argc!=3)throw std::runtime_error("usage: soma-softmax-test TABLE CUDA_FIXTURE");
 load_soma_exp2_table(argv[1]);std::ifstream f(argv[2],std::ios::binary);
 auto read=[&](auto &value){f.read(reinterpret_cast<char *>(&value),sizeof(value));};
 uint32_t count=0;read(count);
 if(count!=24)throw std::runtime_error("invalid softmax fixture layer count");size_t differences=0,half=0,values=0;
 for(uint32_t layer=0;layer<count;++layer){
  uint32_t width,rows;read(width);read(rows);if(!f || !width || width>1024 || !rows || rows>4096)throw std::runtime_error("invalid softmax fixture shape");std::vector<float> x(width*rows),expected(x.size()),actual(x.size());
  f.read(reinterpret_cast<char *>(x.data()),x.size()*4);f.read(reinterpret_cast<char *>(expected.data()),expected.size()*4);
  if(!f)return 3;
  for(size_t row=0;row<rows;++row)soma_warp_softmax(x.data()+row*width,actual.data()+row*width,width);
  for(size_t i=0;i<x.size();++i){differences+=actual[i]!=expected[i];half+=ggml_fp32_to_fp16(actual[i])!=ggml_fp32_to_fp16(expected[i]);}
  values+=x.size();
 }
 std::cout<<"values="<<values<<" float_mismatches="<<differences<<" half_mismatches="<<half<<'\n';
 if(values!=93312 || differences!=1 || half)throw std::runtime_error("softmax calibration regression");
 if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing fixture bytes");
 return 0;
} catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
