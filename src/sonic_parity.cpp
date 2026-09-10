#include <motionbricks/sonic.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char ** argv) {
    try {
        if(argc<4 || argc>6) throw std::runtime_error("usage: motionbricks-sonic-parity MODEL.gguf FIXTURE.mbsonic cpu|vulkan [THREADS [--report-all]]");
        const bool report_all=argc==6 && std::string(argv[5])=="--report-all";
        if(argc==6 && !report_all) throw std::runtime_error("invalid report option");
        char error[1024]{};
        auto check=[&](mb_status status){ if(status!=MB_OK) throw std::runtime_error(error); };
        mb_runtime_options * raw=nullptr; check(mb_runtime_options_create(&raw,error,sizeof(error)));
        std::unique_ptr<mb_runtime_options,decltype(&mb_runtime_options_free)> options(raw,mb_runtime_options_free);
        if(std::string(argv[3])!="cpu" && std::string(argv[3])!="vulkan") throw std::runtime_error("invalid backend");
        check(mb_runtime_options_set_device(raw,std::string(argv[3])=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,error,sizeof(error)));
        const std::string thread_arg=argc>=5?argv[4]:"4";
        if(thread_arg.empty() || thread_arg.find_first_not_of("0123456789")!=std::string::npos)
            throw std::runtime_error("invalid thread count");
        const auto threads=std::stoul(thread_arg);
        if(threads<1 || threads>1024) throw std::runtime_error("thread count must be 1..1024");
        check(mb_runtime_options_set_threads(raw,static_cast<uint32_t>(threads),error,sizeof(error)));
        mb_sonic * pointer=nullptr; check(mb_sonic_load(argv[1],raw,&pointer,error,sizeof(error)));
        std::unique_ptr<mb_sonic,decltype(&mb_sonic_free)> model(pointer,mb_sonic_free);
        std::ifstream f(argv[2],std::ios::binary); std::array<char,8> magic{}; uint32_t samples=0;
        f.read(magic.data(),8); f.read(reinterpret_cast<char*>(&samples),4);
        if(!f || std::memcmp(magic.data(),"MBSONIC1",8)!=0 || samples==0 || samples>20000) throw std::runtime_error("invalid fixture header");
        auto read=[&](size_t n){std::vector<float> a(n); if(!f.read(reinterpret_cast<char*>(a.data()),static_cast<std::streamsize>(n*4))) throw std::runtime_error("truncated fixture"); return a;};
        double maximum=0,relative=0; uint32_t sample=0;
        uint64_t token_mismatches=0;
        std::map<std::string,uint32_t> failures;
        auto compare=[&](const std::string & name,const std::vector<float> & a,const std::vector<float> & b,bool exact=false){
            double mx=0,ss=0,den=0; for(size_t i=0;i<a.size();++i) {
                if(!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("non-finite "+name);
                double delta=double(a[i])-b[i]; mx=std::max(mx,std::abs(delta)); ss+=delta*delta; den+=double(b[i])*b[i];
                if(exact && delta!=0) ++token_mismatches;
            }
            double rel=std::sqrt(ss/std::max(den,1e-20)); maximum=std::max(maximum,mx); relative=std::max(relative,rel);
            // Declared before evaluating captures: F32 MLP abs and relative L2
            // both required; discrete FSQ codes require exact equality.
            if(mx>(exact?0:5e-4) || (!exact && rel>1e-5)) {
                if(report_all) ++failures[name];
                else throw std::runtime_error("sample "+std::to_string(sample)+" "+name+" max="+std::to_string(mx)+" relative="+std::to_string(rel));
            }
        };
        const auto start=std::chrono::steady_clock::now();
        for(;sample<samples;++sample) {
            auto enc=read(1762),dec=read(994); std::vector<float> tokens(64),actions(29);
            check(mb_sonic_encode(pointer,enc.data(),enc.size(),tokens.data(),tokens.size(),error,sizeof(error)));
            for(uint32_t i=0;i<5;++i) {
                uint64_t n=0; check(mb_sonic_layer(pointer,1,i,nullptr,0,&n,error,sizeof(error))); std::vector<float> layer(n);
                check(mb_sonic_layer(pointer,1,i,layer.data(),n,&n,error,sizeof(error))); compare("encoder."+std::to_string(i),layer,read(n));
            }
            compare("tokens",tokens,read(64),true);
            check(mb_sonic_decode(pointer,dec.data(),dec.size(),actions.data(),actions.size(),error,sizeof(error)));
            for(uint32_t i=0;i<7;++i) {
                uint64_t n=0; check(mb_sonic_layer(pointer,0,i,nullptr,0,&n,error,sizeof(error))); std::vector<float> layer(n);
                check(mb_sonic_layer(pointer,0,i,layer.data(),n,&n,error,sizeof(error))); compare("decoder."+std::to_string(i),layer,read(n));
            }
            compare("actions",actions,read(29));
            std::copy(tokens.begin(),tokens.end(),dec.begin());
            check(mb_sonic_decode(pointer,dec.data(),dec.size(),actions.data(),actions.size(),error,sizeof(error)));
            compare("composed_actions",actions,read(29));
        }
        if(f.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture data");
        std::cout<<(failures.empty()?"PASS":"FAIL")<<" samples="<<samples<<" backend="<<argv[3]<<" threads="<<threads<<" max_abs="<<maximum<<" max_relative_l2="<<relative
                 <<" token_mismatches="<<token_mismatches
                 <<" elapsed_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
        for(const auto & [name,count]:failures) std::cout<<name<<" failed_samples="<<count<<"\n";
        return failures.empty()?0:1;
    } catch(const std::exception & e) { std::cerr<<e.what()<<"\n"; return 1; }
}
