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

class SkyNetOS
{
    public:
        static ISkyNetClassificationProtocol* LoadModule(std::string moduleName, void** plibHandle); 
        static void ReleaseModule(ISkyNetClassificationProtocol** pModule,void** pLibHandle);
        static unsigned int getPID();
        static bool CreateDirectory(const std::string& dirname);
        static std::string GetHomeDirectory(void);
};



#endif //__OS_INC__
