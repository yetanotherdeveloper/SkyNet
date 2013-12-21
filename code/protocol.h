#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
#include <string>

// Note: Module should export following function:
//ISkyNetClassificationProtocol* CreateModule();

class ISkyNetClassificationProtocol
{
    public:
        virtual ~ISkyNetClassificationProtocol() {};
        virtual void Run(void) = 0;
        virtual void About()   = 0;
        std::string Identify(){ return std::string("ISkyNetClassificationProtocol");}
};
#endif //__SKYNET_PROTOCOL__
