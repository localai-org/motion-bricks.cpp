#include "soma_welford.hpp"
#include <vector>
#include <iostream>
using namespace motionbricks::detail;
int main() {
    for(const std::size_t width:{512U,1024U}) {
        std::vector<float> input(width,3.25F),scale(width,2),bias(width,0.125F),output(width);
        const auto constant=soma_welford_statistics(input.data(),width);
        if(constant.mean!=3.25F||constant.variance!=0)return 1;
        soma_welford_normalize(input.data(),scale.data(),bias.data(),output.data(),width,1e-5F);
        for(auto value:output)if(value!=0.125F)return 2;
        for(std::size_t i=0;i<width;++i)input[i]=i%2?6.0F:2.0F;
        const auto symmetric=soma_welford_statistics(input.data(),width);
        if(symmetric.mean!=4||symmetric.variance!=4)return 3;
    }
    float value=0;
    try { soma_welford_statistics(&value,1); }
    catch(const std::invalid_argument &) { return 0; }
    std::cerr<<"unsupported width accepted\n";return 4;
}
