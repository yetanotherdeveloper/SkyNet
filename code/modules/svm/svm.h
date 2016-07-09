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
    const std::vector<float> & RunCL(const std::vector<std::vector<float>> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    void RunRef( const std::vector< std::vector<float> > &trainingData,
                 const std::vector<int> &trainingLabels,
                 const std::vector<std::vector<float>>   &validationData,
                 const std::vector<int> &validationLabels,
                 unsigned int max_iterations,
                 SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter);
    void setWeights(std::vector< float > &initial_weights);
    float getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels);
    std::vector<int> & getClassification(const std::vector<std::vector<float>> & data);
    void About();
    const std::string About() const;
    void reshape(unsigned int num_inputs, unsigned int num_categories);
};
#endif //__SVM__

