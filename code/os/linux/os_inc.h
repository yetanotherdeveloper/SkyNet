#ifndef __OS_INC__
#define __OS_INC__

#include <cstdio>

#ifdef DEBUG
#define SKYNET_DEBUG(...) printf("(%s,%d) DEBUG: ",__FILE__, __LINE__); printf( __VA_ARGS__ )
//#define SKYNET_DEBUG(...) printf(__VA_ARGS__ )
#else  // Here is for Release
#define SKYNET_DEBUG(...) 0
#endif  //DEBUG


#define SKYNET_INFO(...) printf( __VA_ARGS__ )

#include <string>
class ISkyNetClassificationProtocol;

namespace cl
{
    class Device;
}

class SkyNetOS
{
    public:
        static ISkyNetClassificationProtocol* LoadModule(std::string moduleName, void** plibHandle,const cl::Device *const pdevice); 
        static void ReleaseModule(ISkyNetClassificationProtocol** pModule,void** pLibHandle);
};



#endif //__OS_INC__
