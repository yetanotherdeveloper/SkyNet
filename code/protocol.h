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
    virtual void RunCL(void)                                                                          = 0;
    virtual const std::vector<float> & RunRef(const std::vector<point> &, const std::vector<float> &) = 0;
    virtual const std::string About() const                                                           = 0;
    virtual bool makeDiagnostic()                                                                     = 0;
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


class SkyNetDiagnostic
{
private:
    std::vector<std::vector<float> > m_historyOfWeights;    /// Weights assuming that in chronological
                                                            /// order First one is the oldest(earliest) iteration
    std::string m_dumpDirName;                              /// Directory where gathered/generated data will be stored
public:
    SkyNetDiagnostic(const std::string & partialName);
    ~SkyNetDiagnostic();
    void storeWeights(const std::vector<float> &weights);
    void dumpWeights();
};
#endif //__SKYNET_PROTOCOL__

