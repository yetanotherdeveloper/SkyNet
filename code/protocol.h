#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
#include <string>
#include <memory>
#include <CL/cl.hpp>

// Note: Module should export following function:
//extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)

class ISkyNetClassificationProtocol
{
    public:
        virtual ~ISkyNetClassificationProtocol() {};
        virtual void Run(void) = 0;
        virtual const std::string About() const  = 0;
        std::string Identify(){ return std::string("ISkyNetClassificationProtocol");}
};

class SkyNetOpenCLHelper
{
    private:
        static cl_int err;
    public:
        static std::unique_ptr<cl::Context> createCLContext(const cl::Device* pdevice);
        static std::unique_ptr<cl::CommandQueue> createCLCommandQueue( const cl::Context& context, const cl::Device& device);
};

#endif //__SKYNET_PROTOCOL__
