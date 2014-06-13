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
    std::unique_ptr< cl::Context >      m_pContext;
    std::unique_ptr< cl::CommandQueue > m_pCommandQueue;
    std::unique_ptr< cl::Kernel >       m_plaKernel;
    const cl::Device *const             m_pdevice;
    std::vector< float >                m_weights;
public:
    StochasticGradientDescent( const cl::Device * const pdevice );
    ~StochasticGradientDescent();
    const std::vector<float> & RunCL(const std::vector<point> &, const std::vector<float> &, SkyNetDiagnostic &);
    const std::vector< float > & RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,SkyNetDiagnostic &diagnostic);              
    const std::string About() const;
    static std::string composeAboutString( const cl::Device *const pdevice );
private:
    float  classifyPoint( const point &rpoint );
    float getPartialErrorDerivative( const point &rpoint );
    bool updateWeights(const point &randomSample);
};
#endif //__SGD__

