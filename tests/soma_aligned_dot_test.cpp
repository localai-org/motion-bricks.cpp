#include "soma_aligned_dot.hpp"
#include <array>
#include <cmath>
#include <iostream>
using motionbricks::detail::soma_aligned_dot;
int main() {
    std::array<float,20> a{},b{};b.fill(1);
    for(int i=0;i<4;++i)a[i]=b[i]=std::ldexp(1.0F,-13);
    a[4]=1;a[5]=-1;
    if(soma_aligned_dot(a.data(),b.data(),20)!=std::ldexp(1.0F,-24) ||
       soma_aligned_dot(a.data(),b.data(),20,false)!=0) {
        std::cerr<<"short remainder must precede cancellation block\n";return 1;
    }
    a.fill(0);b.fill(1);a[0]=1;
    for(int i=1;i<4;++i)a[i]=b[i]=std::ldexp(1.0F,-12);
    const float expected=1+std::ldexp(1.0F,-23);
    if(soma_aligned_dot(a.data(),b.data(),4)!=expected)return 2;
    for(auto & value:a)value=-value;
    if(soma_aligned_dot(a.data(),b.data(),4)!=-expected)return 3;
    if(soma_aligned_dot(a.data(),b.data(),0)!=0)return 4;
    return 0;
}
