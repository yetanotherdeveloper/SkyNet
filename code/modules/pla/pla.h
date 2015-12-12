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
    std::vector<float>                m_weights;
    std::vector<int>                  m_classification;   /// Last results of classification query (getClassification)
public:
    PerceptronLearningAlgorithm();
    ~PerceptronLearningAlgorithm();
    void RunCL();
    void setWeights(std::vector< float > &initial_weights);
    const std::vector<float> & RunCL(const std::vector<std::vector<float>> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);

void RunRef( const std::vector< std::vector<float> > &trainingData,
             const std::vector<int> &trainingLabels,
             const std::vector<std::vector<float>>   &validationData,
             const std::vector<int> &validationLabels,
             SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter);

    const std::string About() const;
    static std::string composeAboutString();
    float getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels);

    std::vector<int> & getClassification(const std::vector<std::vector<float>> & data);
private:
    int  classifyPoint(const std::vector<float> &rpoint);
    float getSampleClassificationError( const int sample, float output );
    void updateWeights(const std::vector<float>& rpoint, const int classification);
    bool getMisclassifiedPoint(const std::vector<std::vector<float>> & trainingData, 
                               const std::vector<int> &trainingLabels, 
                               const std::vector<float>** output);

    bool getMisclassifiedPoint(const std::vector<std::vector<float>> & trainingData, 
                               const std::vector<int> &trainingLabels,
                               const std::vector<float>** outputData,
                               const int** outputLabel);
};
#endif //__PLA__

