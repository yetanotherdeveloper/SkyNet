#ifndef __PLA__
#define __PLA__
#include <string>
#include "protocol.h"

namespace cl
{
class Device;
class Context;
class CommandQueue;
class Kernel;
}

class PerceptronLearningAlgorithm : public ISkyNetClassificationProtocol
{
private:
    std::string                       m_about;
    std::unique_ptr<cl::Context>      m_pContext;
    std::unique_ptr<cl::CommandQueue> m_pCommandQueue;
    std::unique_ptr<cl::Kernel>       m_plaKernel;
    const cl::Device *const           m_pdevice;
    std::vector<float>                m_weights;
    std::vector<int>                  m_classification;   /// Last results of classification query (getClassification)
public:
    PerceptronLearningAlgorithm(const cl::Device * const pdevice);
    ~PerceptronLearningAlgorithm();
    void RunCL();
    void setWeights(std::vector< float > &initial_weights);
    const std::vector<float> & RunCL(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    const std::vector<float> & RunRef(const std::vector<point> & trainingData,const std::vector<point> &validationData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    const std::string About() const;
    static std::string composeAboutString(const cl::Device* const pdevice);
    float getError(const std::vector<point> & data);
    std::vector<int> & getClassification(const std::vector<point> & data);
private:
    int  classifyPoint(const point &rpoint);
    float getSampleClassificationError( const point& sample, float output );
    void updateWeights(const point& rpoint);
    bool getMisclassifiedPoint(const std::vector<point> & trainingData, const point** output);
};
#endif //__PLA__

