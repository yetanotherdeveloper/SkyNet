#ifndef __SGD
#define __SGD__
#include <string>
#include "protocol.h"

namespace cl
{
class Device;
class Context;
class CommandQueue;
class Kernel;
}

class StochasticGradientDescent : public ISkyNetClassificationProtocol
{
private:
    std::string                         m_about;
    float                               m_theta; /// learning grade
    float                               m_flatness; /// Value below which sum of weights updates, we declare as flat
    std::vector< float >                m_weights;
    std::vector<int>                    m_classification;   /// Last results of classification query (getClassification)
public:
    StochasticGradientDescent();
    ~StochasticGradientDescent();
    const std::vector<float> & RunCL(const std::vector<std::vector<float>> &, SkyNetDiagnostic &, SkynetTerminalInterface& exitter);
    void RunRef( const std::vector< std::vector<float> > &trainingData,
                 const std::vector<int> &trainingLabels,
                 const std::vector<std::vector<float>>   &validationData,
                 const std::vector<int> &validationLabels,
                 SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter);
    const std::string About() const;
    void setWeights(std::vector< float > &initial_weights);
    static std::string composeAboutString();
    float getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels);
    std::vector<int> & getClassification(const std::vector<std::vector<float>> & data);
private:
    float  classifyPoint( const std::vector<float> &rpoint );
    float getPartialErrorDerivative( const std::vector<float> &rpoint );
    bool updateWeights(const std::vector<float> &randomSample, int label);
};
#endif //__SGD__

