#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
#include <string>
#include <memory>
#include <CL/cl.hpp>

// Note: Module should export following function:
//extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)

struct point {
    float x;                        //! first coord of point
    float y;                        //! second coord of point
    int classification;             //! +1 or -1 meaning the classification area
};

class ISkyNetClassificationProtocol
{
public:
    virtual ~ISkyNetClassificationProtocol() {
    };
    virtual void RunCL(void)                         = 0;
    virtual void RunRef(const std::vector<point> & ) = 0;
    virtual const std::string About() const          = 0;
    std::string Identify()
    {
        return std::string("ISkyNetClassificationProtocol");
    }
};

class SkyNetOpenCLHelper
{
private:
    static cl_int err;
public:
    static std::unique_ptr<cl::Context> createCLContext(const cl::Device* pdevice);
    static std::unique_ptr<cl::CommandQueue> createCLCommandQueue( const cl::Context& context, const cl::Device& device);
    static std::unique_ptr<cl::Kernel> makeKernels(const cl::Context & context,const cl::Device & target_device, const std::string & kernelSource, const std::string kernelName);
};

#endif //__SKYNET_PROTOCOL__
