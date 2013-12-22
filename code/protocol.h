#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
#include <string>

// Note: Module should export following function:
//extern "C" ISkyNetClassificationProtocol* CreateModule()

class ISkyNetClassificationProtocol
{
    public:
        virtual ~ISkyNetClassificationProtocol() {};
        virtual void Run(void) = 0;
        virtual const std::string About() const  = 0;
        std::string Identify(){ return std::string("ISkyNetClassificationProtocol");}
};
#endif //__SKYNET_PROTOCOL__
