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
    const std::vector<float> & RunCL(const std::vector<point> &, SkyNetDiagnostic &, SkynetTerminalInterface& exitter);
    void RunRef(const std::vector<point> & trainingData,const std::vector<point> &validationData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);
    const std::string About() const;
    void setWeights(std::vector< float > &initial_weights);
    static std::string composeAboutString();
    float getError(const std::vector<point> & data);
    std::vector<int> & getClassification(const std::vector<point> & data);
private:
    float  classifyPoint( const point &rpoint );
    float getPartialErrorDerivative( const point &rpoint );
    bool updateWeights(const point &randomSample);
};
#endif //__SGD__

