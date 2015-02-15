#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
#include <iostream>
#include <termios.h>    // Termios 
#include <fcntl.h>      // file control
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
    struct historicalNote
    {
        std::vector<float> m_weights;
        float m_error;
        historicalNote(const std::vector<float> &weights, float error) : m_weights(weights), m_error(error)
        {
        }

    };
private:
    std::vector<historicalNote> m_history;                /// Weights assuming that in chronological
                                                          /// order First one is the oldest(earliest) iteration
    std::string m_dumpDirName;                             /// Directory where gathered/generated data will be stored
public:
    SkyNetDiagnostic();
    ~SkyNetDiagnostic();
    void reset();
    void storeWeightsAndError(const std::vector<float> &weights, float error);
    void makeWeightsAnalysis(const std::string& dirName);
    void saveWeightsToFile(const std::string& dirName);
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

/// This module can check if specific key was pressed 
/// if specific key was pressed then call operator returns true
class SkynetTerminalInterface
{
private:
    char           m_trigger_key;
    struct termios m_original, m_modified;
    int            m_stdio_flags;

public:
    SkynetTerminalInterface(char trigger_key) : m_trigger_key(trigger_key)
    {
        std::cout << "To cease learning and store current result press: " << m_trigger_key << " key." << std::endl;
    }

    bool operator()()
    {
        initTermios();
        char a = getchar();
        resetTermios();
        return a == m_trigger_key ? true : false;
    }

private:

    void initTermios(void)
    {
        tcgetattr(0,&m_original);
        m_modified          = m_original;
        m_modified.c_lflag &= ~ICANON;
        m_modified.c_lflag &= ~ECHO;
        tcsetattr(0, TCSANOW,&m_modified);
        m_stdio_flags = fcntl(0,F_GETFL,0);
        fcntl(0,F_SETFL, m_stdio_flags | O_NONBLOCK);
    }

    void resetTermios(void)
    {
        tcsetattr(0, TCSANOW, &m_original);
        fcntl(0,F_SETFL, m_stdio_flags );
    }
};

class ISkyNetClassificationProtocol
{
public:
    virtual ~ISkyNetClassificationProtocol() {
    };
    virtual const std::vector<float> & RunCL(const std::vector<point> &, SkyNetDiagnostic &, SkynetTerminalInterface& ) = 0;
    virtual const std::vector<float> & RunRef(const std::vector<point> &, SkyNetDiagnostic &, SkynetTerminalInterface&) = 0;
    virtual float getError(const std::vector<point> & data)                                                             = 0;
    virtual std::vector<int> & getClassification(const std::vector<point> & data)                                       = 0;
    virtual const std::string About() const                                                                             = 0;
    virtual void setWeights(std::vector< float > &all_weights)                                                          = 0;
    std::string Identify()
    {
        return std::string("ISkyNetClassificationProtocol");
    }
};
#endif //__SKYNET_PROTOCOL__

