#include "soma_rsqrt_table.hpp"
#include <array>
#include <iostream>
#include <limits>
using namespace motionbricks::detail;
int main(int argc,char ** argv) try {
    if(argc!=3)throw std::runtime_error("usage: soma-rsqrt-table-test TABLE CUDA_PAIRS");
    load_soma_rsqrt_table(argv[1]);
    std::ifstream input(argv[2],std::ios::binary);
    if(!input)throw std::runtime_error("cannot read independent CUDA pairs");
    std::array<float,2> pair;std::size_t count=0;
    while(input.read(reinterpret_cast<char *>(pair.data()),sizeof(pair))) {
        if(soma_calibrated_rsqrt(pair[0])!=pair[1])throw std::runtime_error("CUDA rsqrt mismatch");
        ++count;
    }
    if(input.gcount()!=0 || count!=100720)throw std::runtime_error("invalid CUDA pair fixture");
    for(float invalid:{0.0F,-1.0F,std::numeric_limits<float>::denorm_min(),
                       std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected=false;
        try { (void)soma_calibrated_rsqrt(invalid); } catch(const std::runtime_error &) { rejected=true; }
        if(!rejected)throw std::runtime_error("unsupported rsqrt input accepted");
    }
    std::cout<<count<<" independent CUDA instruction outputs matched exactly\n";
    return 0;
} catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
