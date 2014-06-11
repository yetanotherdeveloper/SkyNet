#ifndef __NN
#define __NN__
#include <string>
#include <random>
#include "protocol.h"

namespace cl
{
class Device;
class Context;
class CommandQueue;
class Kernel;
}

class NeuralNetwork : public ISkyNetClassificationProtocol
{

/// Struct representing single layer of Neural Network
struct NeuralLayer
{
/// Struct representing single Neuron
struct Neuron
{
    private:
        static std::uniform_real_distribution< float > s_randFloat;
        static std::random_device s_rd;
        std::vector< float > m_weights;
    public:
        Neuron( unsigned int numInputs );   /// weights initialization
        ~Neuron();
};
public:
    std::vector<Neuron> m_neurons;
    NeuralLayer(unsigned int nrInputs,unsigned int nrNeurons);
    ~NeuralLayer();
};


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
    NeuralNetwork( const cl::Device * const pdevice , unsigned int nrInputs, unsigned int nrLayers);
    ~NeuralNetwork();
    void RunCL();
    const std::vector< float > & RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,SkyNetDiagnostic &diagnostic);              
    const std::string About() const;
    static std::string composeAboutString( const cl::Device *const pdevice );
private:
    float  classifyPoint( const point &rpoint );
    float getPartialErrorDerivative( const point &rpoint );
    bool updateWeights(const point &randomSample);
};
#endif //__NN__

