// ELF symbol interposition models an incompatible loaded SDK without ever
// allocating an ABI-dependent mjModel/mjData or loading external assets.
#include <motionbricks/physics.h>
#include <cassert>
#include <cstring>

extern "C" int mj_version() { return -1; }

int main() {
    char version[32]{},error[128]{};
    assert(mb_physics_engine_version(version,sizeof(version),error,sizeof(error))==MB_INCOMPATIBLE_MODEL);
    assert(version[0]==0 && std::strstr(error,"ABI mismatch"));
    // An opaque stand-in is safe here only because rejection must precede
    // any access to the policy, config, scene or MuJoCo struct layout.
    alignas(void*) char stand_in[sizeof(void*)]{};
    auto * sonic=reinterpret_cast<mb_sonic*>(stand_in);
    mb_physics * session=nullptr;
    assert(mb_physics_create(sonic,"must-not-load.xml","must-not-load.config",&session,error,sizeof(error))==MB_INCOMPATIBLE_MODEL);
    assert(!session && std::strstr(error,"ABI mismatch"));
}
