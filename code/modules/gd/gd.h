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

class GradientDescent : public ISkyNetClassificationProtocol
{
private:
    std::string                         m_about;
    float                               m_theta; /// learning grade
    std::unique_ptr< cl::Context >      m_pContext;
    std::unique_ptr< cl::CommandQueue > m_pCommandQueue;
    std::unique_ptr< cl::Kernel >       m_plaKernel;
    const cl::Device *const             m_pdevice;
    std::vector< float >                m_weights;
public:
    GradientDescent( const cl::Device * const pdevice );
    ~GradientDescent();
    void RunCL();
    const std::vector< float > & RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,SkyNetDiagnostic &diagnostic);              
    const std::string About() const;
    static std::string composeAboutString( const cl::Device *const pdevice );
private:
    float  classifyPoint( const point &rpoint );
    float getPartialErrorDerivative( const point &rpoint );
    bool updateWeights(const std::vector<point> & trainingData);
};
#endif //__PLA__

