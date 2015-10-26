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
    std::vector< float >    m_weights;
    std::vector< int >      m_classification;                           /// Last results of classification query  (getClassification)
public:
    SupportVectorMachine();
    ~SupportVectorMachine();
    const std::vector<float> & RunCL(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    void RunRef(const std::vector<point> & trainingData, const std::vector<point> &validationData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    void setWeights(std::vector< float > &initial_weights);
    float getError(const std::vector<point> & data);
    std::vector<int> & getClassification(const std::vector<point> & data);
    void About();
    const std::string About() const;
};
#endif //__SVM__

