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


class SkyNetDiagnostic
{
private:
    std::vector<std::vector<float> > m_historyOfWeights;    /// Weights assuming that in chronological
                                                            /// order First one is the oldest(earliest) iteration
    std::string m_dumpDirName;                              /// Directory where gathered/generated data will be stored
public:
    SkyNetDiagnostic();
    ~SkyNetDiagnostic();
    void storeWeights(const std::vector<float> &weights);
    void dumpWeights(const std::string& dirName);
    void makeTrainingAnalysis(const std::string& dirName,const std::vector<point> & set,
                              const std::vector<float> &targetWeights,const std::vector<float> &learnedWeights);
    void makeGeneralizationAnalysis(const std::string& dirName,const std::vector<point> & set,
                              const std::vector<float> &targetWeights,const std::vector<float> &learnedWeights);
private:
    void makeAnalysis(const std::string& dirName,const std::string& dataFilename, const std::string& scriptFilename,
                                            const std::vector<point> & set, const std::vector<float> &targetWeights,
                                            const std::vector<float> &learnedWeights);
};

typedef void (SkyNetDiagnostic::*fp_storeWeights)(const std::vector<float> &weights);

class SkyNetOpenCLHelper
{
private:
    static cl_int err;
public:
    static std::unique_ptr<cl::Context> createCLContext(const cl::Device* pdevice);
    static std::unique_ptr<cl::CommandQueue> createCLCommandQueue( const cl::Context& context, const cl::Device& device);
    static std::unique_ptr<cl::Kernel> makeKernels(const cl::Context & context,const cl::Device & target_device, const std::string & kernelSource, const std::string kernelName);
};


class ISkyNetClassificationProtocol
{
public:
    virtual ~ISkyNetClassificationProtocol() {
    };
    virtual void RunCL(void)                                                                                               = 0;
    virtual const std::vector<float> & RunRef(const std::vector<point> &, const std::vector<float> &, SkyNetDiagnostic & ) = 0;
    virtual const std::string About() const                                                                                = 0;
    std::string Identify()
    {
        return std::string("ISkyNetClassificationProtocol");
    }
};
#endif //__SKYNET_PROTOCOL__

