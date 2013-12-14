#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
class ISkyNetClassificationProtocol
{
    public:
        virtual ~ISkyNetClassificationProtocol() {};
        virtual void Run(void) = 0;
        virtual void About()   = 0;
};
#endif //__SKYNET_PROTOCOL__
