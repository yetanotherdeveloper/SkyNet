#ifndef __SVM__
#define __SVM__
#include <string>
#include "protocol.h"

namespace cl
{
    class Device;
    class Context;
    class CommandQueue;
}

class SupportVectorMachine : public ISkyNetClassificationProtocol
{
private:
    std::string             m_about;
    cl::Context             *m_context;
    cl::CommandQueue        *m_queue;
    const cl::Device *const m_pdevice;
    std::vector< float >    m_weights;
    std::vector< int >      m_classification;                           /// Last results of classification query  (getClassification)
public:
    SupportVectorMachine(const cl::Device * const pdevice);
    ~SupportVectorMachine();
    const std::vector<float> & RunCL(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    const std::vector<float> & RunRef(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    float getError(const std::vector<point> & data);
    std::vector<int> & getClassification(const std::vector<point> & data);
    void About();
    const std::string About() const;
};
#endif //__SVM__

