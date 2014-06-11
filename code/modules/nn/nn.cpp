#include "nn.h"
#include <CL/cl.hpp>
#include <random>
#include <cmath>
#include <limits>

static std::string kernelSource = "__kernel void dodaj(float veciu) \
                                   {  \
                                            veciu = 1.0f; \
                                   }";

extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)
{
    return new NeuralNetwork(pdevice,2,2);  // Just for testing, architecture depends heavily on problem we are to solve
}

/*! Build kernels , initialize data
 *
 */
NeuralNetwork::NeuralNetwork(const cl::Device* const pdevice, unsigned int nrInputs, unsigned int nrLayers) : m_about(NeuralNetwork::composeAboutString(pdevice) ), m_pdevice(pdevice),
                                                     m_theta(0.1), m_flatness(0.0000001f)
{
    cl_int err;

    // TODO:
    // - create command queue
    // - build programs, make kernels
    // - store device, command queue, context
    m_pContext = SkyNetOpenCLHelper::createCLContext(pdevice);

    std::vector<cl::Device> context_devices;
    m_pContext->getInfo(CL_CONTEXT_DEVICES,&context_devices);

    m_pCommandQueue = SkyNetOpenCLHelper::createCLCommandQueue( *m_pContext, *pdevice);

    m_plaKernel = SkyNetOpenCLHelper::makeKernels(*m_pContext, *pdevice, kernelSource, "dodaj" );

    //TODO: error handling section


    // Create Neural Network infrastructure
    /* Here we create topology of Neural Network. This will be adjusted in the future
     *  regarding the probem we use Neural Network to solve 
     *
     *  So this is just example NN topology, looking like this:
     *     N      <-- output layer
     *    / \
     *   N   N    <-- first layer
     *
     */
    for( unsigned int i = 0; i < nrLayers; ++i )
    {
        m_layers.push_back(NeuralLayer(nrInputs,(unsigned int)powf(2.0f,(float)(nrLayers - i -1)),NeuronFlags::INIT_ONE));
    }

}


std::string NeuralNetwork::composeAboutString(const cl::Device* const pdevice)
{
    std::string aboutString;
    pdevice->getInfo(CL_DEVICE_NAME, &aboutString);
    aboutString.insert(0,"Neural Network Algorithm (");
    aboutString.append(")");
    return aboutString;
}

// Get Error based on current state of neural network



/*! Function updating weights based on current stochastic gradient descent
 *  Desc: 
 *          Update rule: w_k+1 <-- w_k - theta*grad(E_in(w(k))          
 *          E_in(w(t)) = (x*w(t) - target_value )**2
 *          dE_in/dw(t) = 2*(x*w(t) - target_value)
 *
 *          where:
 *            x*w(t) is learned_value
 *            E_in(w(t)) is square error eg. square error on random sample() from training set
 *
 */ 
bool NeuralNetwork::updateWeights(const point &randomSample)
{
    // TODO: make it working for any dimentions not just two
    // Perform gradient descent on given (assuming random) sample
    float dw0,dw1,dw2;
    dw0 = -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification);                // gradient per w0
    dw1 = dw0*randomSample.x; // gradient per w1
    dw2 = dw0*randomSample.y; // gradient per w2

    m_weights[0] += dw0;
    m_weights[1] += dw1;
    m_weights[2] += dw2;

    // sum of updates to weights is less then our flatness value then
    // we decalre that no progress is made
    if(dw0*dw0 + dw1*dw1 + dw2*dw2 <= m_flatness) {
        return true;
    }

    //m_weights[0] += -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification);                // gradient per w0
    //m_weights[1] += -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification)*randomSample.x; // gradient per w1
    //m_weights[2] += -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification)*randomSample.y; // gradient per w2
    return false;
}


// TODO: move this constant to some other area or make it derived based on number of training  points
const std::vector<float> & NeuralNetwork::RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,SkyNetDiagnostic &diagnostic)              
{
    std::uniform_int_distribution< int > sample_index( 0, trainingData.size() -1 );
    std::random_device rd;

    m_weights = initial_weights;
    const int max_iterations = 1000*trainingData.size();
    int i=0;
    bool finish = false;
    diagnostic.storeWeights(m_weights);
    while((i<max_iterations)&&(finish == false))  {
        finish = updateWeights(trainingData[sample_index(rd)]);
        diagnostic.storeWeights(m_weights);
        if(finish == false) {
            ++i;
        } else {
            // TODO: If flatness was reached then
            // check if error is below certain error threshold 
        }

    } 
    if(finish == false) {
        printf("Warning: Stochastic Gradient Descent Learning alogorithm exhusted all iterations. TODO: Make proper termination criteria\n");
    }
    return m_weights;
}


void NeuralNetwork::RunCL()
{
    float testValue = 0.0f;
    m_plaKernel->setArg(0,&testValue);
    m_pCommandQueue->enqueueTask(*m_plaKernel);
}


const std::string NeuralNetwork::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

NeuralNetwork::~NeuralNetwork()
{
}

// ----> NeuralLayer struct implementation
/*! Create layer of neurons
 */
const float NeuralNetwork::NeuralLayer::Neuron::minRandValue = -1000000.0f;
const float NeuralNetwork::NeuralLayer::Neuron::maxRandValue = 1000000.0f;
std::uniform_real_distribution< float > NeuralNetwork::NeuralLayer::Neuron::s_randFloat = std::uniform_real_distribution<float>( Neuron::minRandValue, Neuron::maxRandValue);
std::random_device NeuralNetwork::NeuralLayer::Neuron::s_rd; 

NeuralNetwork::NeuralLayer::NeuralLayer( unsigned int nrInputs, unsigned int nrNeurons, NeuronFlags flags )
{
    for( unsigned int i = 0; i < nrNeurons; ++i )
    {
        m_neurons.push_back( Neuron( nrInputs, flags ) );
    }
}

// NeuralLayer struct implementation
NeuralNetwork::NeuralLayer::~NeuralLayer()
{
}


// ----> Neuron struct implementation
/*! Create a neuron with number of weights
 *  equal to number of inputs + 1 (threshold)
 */
NeuralNetwork::NeuralLayer::Neuron::Neuron( unsigned int numInputs, NeuronFlags flags )
{

    // Number of weights is equal to number of inputs + bias (threshold)
    for( unsigned int i = 0; i < numInputs + 1; ++i )
    {
        switch( flags )
        {
        case NeuronFlags::INIT_RANDOM:
            m_weights.push_back( s_randFloat( s_rd ) );
            break;
        case NeuronFlags::INIT_ZERO:
            m_weights.push_back( 0.0f );
            break;
        case NeuronFlags::INIT_ONE:
            m_weights.push_back( 1.0f );
            break;
        }
    }
}


NeuralNetwork::NeuralLayer::Neuron::~Neuron()
{
}


