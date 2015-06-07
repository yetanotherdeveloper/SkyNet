#include "nn.h"
#include <CL/cl.hpp>
#include <random>
#include <cmath>
#include <limits>
#include <cassert>

static std::string kernelSource =
    "__kernel void dodaj(float veciu) \
                                   {  \
                                            veciu = 1.0f; \
                                   }";

extern "C" ISkyNetClassificationProtocol *CreateModule( const cl::Device *const pdevice )
{
    return new NeuralNetwork( pdevice, 2, 2, GradientDescentType::STOCHASTIC );  // Just for testing, architecture depends heavily on problem we are to
                                                // solve
}

/*! Build kernels , initialize data
 *
 */
NeuralNetwork::NeuralNetwork( const cl::Device *const pdevice, unsigned int nrInputs, unsigned int nrLayers, GradientDescentType gdtype) : 
                              m_about( NeuralNetwork::composeAboutString( pdevice ) ), m_pdevice( pdevice ), m_gradType(gdtype)
{
    cl_int err;

    // TODO:
    // - create command queue
    // - build programs, make kernels
    // - store device, command queue, context
    m_pContext = SkyNetOpenCLHelper::createCLContext( pdevice );

    std::vector< cl::Device > context_devices;
    m_pContext->getInfo( CL_CONTEXT_DEVICES, &context_devices );

    m_pCommandQueue = SkyNetOpenCLHelper::createCLCommandQueue( *m_pContext, *pdevice );

    m_plaKernel = SkyNetOpenCLHelper::makeKernels( *m_pContext, *pdevice, kernelSource, "dodaj" );

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
        m_layers.push_back( NeuralLayer( nrInputs,
                                         ( unsigned int )powf( 2.0f,
                                                               ( float )(nrLayers - i - 1) ),
                                         NeuronFlags::INIT_RANDOM ) );
    }

}


std::string NeuralNetwork::composeAboutString( const cl::Device *const pdevice )
{
    std::string aboutString;
    pdevice->getInfo( CL_DEVICE_NAME, &aboutString );
    aboutString.insert( 0, "Neural Network Algorithm (" );
    aboutString.append( ")" );
    return aboutString;
}


std::vector<int> & NeuralNetwork::getClassification(const std::vector<point> & data)
{
    m_classification.clear();
    // each data point has corressponding classification info
    // so we can reserve space upfront
    m_classification.reserve(data.size() );

    // Send each point through NN and get classification error for it
    // later on all errors are summed up and divided by number of samples
    for( unsigned int k = 0; k < data.size(); ++k )
    {
        m_classification.push_back(getNetworkOutput(data[k]) > 0.0f ? 1 : -1);
    }
    return m_classification;
}


/*! Given sample classification error
 *  Square error is used as error measure eg
 *  (NN_classification_value - sample_classification )^2
 */
float NeuralNetwork::getSampleClassificationError( const point& sample, float output )
{
    return powf( (output - ( float )sample.classification), 2.0f );
}


/*! Get all weights from all neurons of all layers and
 *  put it in given container
 */
void NeuralNetwork::getAllWeights(std::vector< float > &all_weights)
{
    all_weights.clear();
    for( unsigned int i = 0; i < m_layers.size(); ++i )
    {
        for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
        {
            for(unsigned int w = 0; w < m_layers[i].m_neurons[j].getWeightsQuantity(); ++w) {
                all_weights.push_back(m_layers[i].m_neurons[j].getWeight(w));
            }
        }
    }
}


// Get Error based on current state of neural network
float NeuralNetwork::getError( const std::vector< point > & data )
{
    // TODO: adjust capacity
    float total_error = 0.0f;

    // Send each point through NN and get classification error for it
    // later on all errors are summed up and divided by number of samples
    for( unsigned int k = 0; k < data.size(); ++k )
    {
        total_error += getSampleClassificationError( data[k], getNetworkOutput(data[k]) );
    }
    return total_error / ( float )data.size();

}


float NeuralNetwork::getNetworkOutput(const point &randomSample)
{
    std::vector< float > input;
    std::vector< float > output;

    // First Layer takes data as input
    for( unsigned int j = 0; j < m_layers[0].m_neurons.size(); ++j )
    {
        input.push_back( m_layers[0].m_neurons[j].getOutput( randomSample ) );
    }

    // hidden layers
    for( unsigned int i = 1; i < m_layers.size(); ++i )
    {
        for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
        {
            output.push_back( m_layers[i].m_neurons[j].getOutput( input ) );
        }
        input = output;
    }
    // Here output should be just a single float number
    assert( output.size() == 1 );

    return output[0];
}
//////////////////////////////////////////////////////////////////////////////////
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
bool NeuralNetwork::updateWeights( const point &randomSample )
{
    // Get final delta: de/ds^l
    std::vector< float >             input;
    std::vector< float >             output;
    std::vector<std::vector<float> > neurons_outputs;    // All outputs are here 0 - first layer, 1 - first hidden..
    // init vector of layers with vector of outputs for each layer
    for(unsigned int m = 0; m < m_layers.size(); ++m) {
        neurons_outputs.push_back(input);
    }

    // FORWARD PROPAGATE

    // First Layer takes data as input
    for( unsigned int j = 0; j < m_layers[0].m_neurons.size(); ++j )
    {
        input.push_back( m_layers[0].m_neurons[j].getOutput( randomSample ) );
    }
    neurons_outputs[0] = input;

    // hidden layers
    for( unsigned int i = 1; i < m_layers.size(); ++i )
    {
        for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
        {
            output.push_back( m_layers[i].m_neurons[j].getOutput( input ) );
        }
        input              = output;
        neurons_outputs[i] = input;
    }
    // Here output should be just a single float number
    assert( output.size() == 1 );
    // Set Final(top level neuron) delta: 2*(tanh(s) - y)*(1 - tanh**2(s))
    // d(tanh(s))/ds = 1 - tanh**2(s)
    m_layers[m_layers.size() - 1].m_neurons[0].setDelta( 2.0f*( output[0] - (float)randomSample.classification )*(1.0f - output[0] * output[0]) );

    // BACKPROPAGATE delta backwards (lower NN layers)
    // starting from previous to highest layer
    for( int l = m_layers.size() - 2; l >= 0; --l )
    {
        // neurons we are to get deltas for
        for( unsigned int n = 0; n < m_layers[l].m_neurons.size(); ++n )
        {
            // For all neurons (of higher layer) attached to analyzed neuron
            float delta = 0.0f;
            for( unsigned int i = 0; i < m_layers[l + 1].m_neurons.size(); ++i )
            {
                // Weight index is an index of considered neuron + 1 as weights 0 is threshold
                delta += m_layers[l + 1].m_neurons[i].getDelta() * m_layers[l + 1].m_neurons[i].getWeight( n + 1 );
            }
            delta *= (1.0f - m_layers[l].m_neurons[n].getOutput() * m_layers[l].m_neurons[n].getOutput() );
            m_layers[l].m_neurons[n].setDelta( delta );
        }
    }


    //TODO : finish can be used to check if enough big update of weights was made 
    //
    // Finish rule is that we end when no single neuron was updated above theta value
    bool finish = true;
    // update first layer
    for( unsigned int j = 0; j < m_layers[0].m_neurons.size(); ++j )
    {
        finish = m_layers[0].m_neurons[j].updateWeights( randomSample ) && finish;
    }

    // update hidden layers
    for( unsigned int i = 1; i < m_layers.size(); ++i )
    {
        for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
        {
            // pass as input to neurons of given layer an output of previous layer
            finish = m_layers[i].m_neurons[j].updateWeights( neurons_outputs[i - 1] ) && finish;
        }
    }
    return finish;
}

/////////////////////////////////////////////////////////////////////////////////
/*! Function updating weights based on current batch gradient descent
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
bool NeuralNetwork::updateWeights(const std::vector< point > & trainingData)
{
    // Get final delta: de/ds^l
    std::vector< float >             input;
    std::vector< float >             output;
    std::vector<std::vector<float> > neurons_outputs;    // All outputs are here 0 - first layer, 1 - first hidden..

    bool finish = true;
    // init vector of layers with vector of outputs for each layer
    for(unsigned int m = 0; m < m_layers.size(); ++m) {
        neurons_outputs.push_back(input);
    }

    // get gradients for every sample and add them together to update weights
    for(unsigned int s = 0; s < trainingData.size(); ++s) {    

        // for every new sample we need to clear outputs of layers
        for(unsigned int m = 0; m< m_layers.size(); ++m) {
            neurons_outputs[m].clear();
        }

        // FORWARD PROPAGATE

        // First Layer takes data as input
        input.clear();
        output.clear();
        for( unsigned int j = 0; j < m_layers[0].m_neurons.size(); ++j )
        {
            input.push_back( m_layers[0].m_neurons[j].getOutput( trainingData[s] ) );
        }
        neurons_outputs[0] = input;

        // hidden layers
        for( unsigned int i = 1; i < m_layers.size(); ++i )
        {
            for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
            {
                output.push_back( m_layers[i].m_neurons[j].getOutput( input ) );
            }
            input              = output;
            neurons_outputs[i] = input;
        }
        // Here output should be just a single float number
        assert( output.size() == 1 );
        // Set Final(top level neuron) delta: 2*(tanh(s) - y)*(1 - tanh**2(s))
        // d(tanh(s))/ds = 1 - tanh**2(s)
        m_layers[m_layers.size() - 1].m_neurons[0].setDelta( 2.0f*( output[0] - (float)trainingData[s].classification )*(1.0f - output[0] * output[0]) );

        // BACKPROPAGATE delta backwards (lower NN layers)
        // starting from previous to highest layer
        for( int l = m_layers.size() - 2; l >= 0; --l )
        {
            // neurons we are to get deltas for
            for( unsigned int n = 0; n < m_layers[l].m_neurons.size(); ++n )
            {
                // For all neurons (of higher layer) attached to analyzed neuron
                float delta = 0.0f;
                for( unsigned int i = 0; i < m_layers[l + 1].m_neurons.size(); ++i )
                {
                    // Weight index is an index of considered neuron + 1 as weights 0 is threshold
                    delta += m_layers[l + 1].m_neurons[i].getDelta() * m_layers[l + 1].m_neurons[i].getWeight( n + 1 );
                }
                delta *= (1.0f - m_layers[l].m_neurons[n].getOutput() * m_layers[l].m_neurons[n].getOutput() );
                m_layers[l].m_neurons[n].setDelta( delta );
            }
        }


        //TODO: This is broken as we should move along average vector not sum of gradients

        // Finish rule is that we end when no single neuron was updated above theta value
        // update first layer
        for( unsigned int j = 0; j < m_layers[0].m_neurons.size(); ++j )
        {
            finish = m_layers[0].m_neurons[j].updateWeights( trainingData[s] ) && finish;
        }

        // update hidden layers
        for( unsigned int i = 1; i < m_layers.size(); ++i )
        {
            for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
            {
                // pass as input to neurons of given layer an output of previous layer
                finish = m_layers[i].m_neurons[j].updateWeights( neurons_outputs[i - 1] ) && finish;
            }
        }


    }   //s

    return finish;
}

/////////////////////////////////////////////////////////////////////////////////////
const std::vector< float > & NeuralNetwork::RunRef( const std::vector< point > &trainingData,
                                                    const std::vector<point>   &validationData,
                                                    SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter)
{
    std::uniform_int_distribution< int > sample_index( 0, trainingData.size() - 1 );
    std::random_device rd;

    std::vector<float> all_weights;
    getAllWeights(all_weights);
    diagnostic.storeWeightsAndError(all_weights,getError(trainingData), getError(validationData) );

    unsigned int max_iterations = 3000;
    //if(m_gradType == GradientDescentType::STOCHASTIC)
    //{
        //max_iterations *= trainingData.size();
    //}

    SkyNetEarlyStop es(max_iterations, 0.4f);

    int       i              = 0;
    bool      finish         = false;
    while( (es.earlyStop(all_weights,getError(validationData)) == false) && (exitter() == false) )
    {
        //float err_before = getError(trainingData);
        if(m_gradType == GradientDescentType::STOCHASTIC)
        {
            finish = updateWeights( trainingData[sample_index( rd )] );
        } else {
            finish = updateWeights( trainingData );
        }

        //float err_after = getError(trainingData);

        getAllWeights(all_weights);
        diagnostic.storeWeightsAndError(all_weights,getError(trainingData), getError(validationData) );
    }
    if( finish == false )
    {
        printf(
            "Warning: Neural Network Learning alogorithm exhusted all iterations. TODO: Make proper termination criteria\n" );
    }
    // TODO: temporayr hack till I can get something decent
    return *(new std::vector<float>(3,0.0f) );
}


const std::vector< float > & NeuralNetwork::RunCL( const std::vector< point > &trainingData,
                                                   SkyNetDiagnostic           &diagnostic,
                                                   SkynetTerminalInterface& exitter)
{
    float testValue = 0.0f;
    m_plaKernel->setArg( 0, &testValue );
    m_pCommandQueue->enqueueTask( *m_plaKernel );
}
///////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::setWeights(std::vector< float > &initial_weights)
{
    // We assume here that given vector of weights correspond in size to
    // architecture of this neural network
    unsigned int k = 0;
    for( unsigned int i = 0; i < m_layers.size(); ++i )
    {
        for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
        {
            for(unsigned int w = 0; w < m_layers[i].m_neurons[j].getWeightsQuantity(); ++w) {
                m_layers[i].m_neurons[j].setWeight(w,initial_weights[k++]);
            }
        }
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////
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
float                                   NeuralNetwork::NeuralLayer::Neuron::s_theta = 0.1f;        /// learning grade
float                                   NeuralNetwork::NeuralLayer::Neuron::s_flatness   = 0.000000000000000001f;
const float                             NeuralNetwork::NeuralLayer::Neuron::minRandValue = -1.0f;
const float                             NeuralNetwork::NeuralLayer::Neuron::maxRandValue = 1.0f;
std::uniform_real_distribution< float > NeuralNetwork::NeuralLayer::Neuron::s_randFloat =
    std::uniform_real_distribution< float >( Neuron::minRandValue, Neuron::maxRandValue );
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
NeuralNetwork::NeuralLayer::Neuron::Neuron( unsigned int numInputs, NeuronFlags flags ) : m_output( 0.0f ), m_delta(
        0.0f )
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


/*! Function to calculate output of single Neuron
 *  Output = tanh(wx)
 *  tanh -- hiperbolic tangent (nonlinear diffrentiable activate function that each neuron needs)
 *  */
float NeuralNetwork::NeuralLayer::Neuron::getOutput( const std::vector< float > & input )
{
    float output = m_weights[0];

    // Input's length has to be equal to weights's length -1 (m_weights[0] is bias)
    for( unsigned int i = 0; i < input.size(); ++i )
    {
        output += m_weights[i + 1] * input[i];
    }
    m_output = ( float )tanh( output );
    return m_output;
}


// This is only for tests operating on point data (input = 2 )
float NeuralNetwork::NeuralLayer::Neuron::getOutput( const point & input )
{
    m_output = ( float )tanh( ( float )(m_weights[0] + m_weights[1] * input.x + m_weights[2] * input.y) );
    return m_output;

}


float NeuralNetwork::NeuralLayer::Neuron::getOutput()
{
    return m_output;
}


void NeuralNetwork::NeuralLayer::Neuron::setDelta( float deltaValue )
{
    m_delta = deltaValue;
}


float NeuralNetwork::NeuralLayer::Neuron::getDelta()
{
    return m_delta;
}

void NeuralNetwork::NeuralLayer::Neuron::setWeight( unsigned int index, float value )
{
    // given index needs to be within range [0..size of mweights-1]
    assert( index < m_weights.size() );
    m_weights[index] = value;
}

float NeuralNetwork::NeuralLayer::Neuron::getWeight( unsigned int index )
{
    // given index needs to be within range [0..size of mweights-1]
    assert( index < m_weights.size() );
    return m_weights[index];
}

float NeuralNetwork::NeuralLayer::Neuron::getWeightsQuantity()
{
    return m_weights.size();
}


//TODO: Make it a template not a copy of functions

bool NeuralNetwork::NeuralLayer::Neuron::updateWeights( const point & input )
{
    float dw0,dw1,dw2;
    // w <-- w - theta * x^(l-1)*Delta^l
    // iterate through all weights of this neuron and update its weights
    dw0 = -this->m_delta * s_theta;
    dw1 = dw0 * input.x;
    dw2 = dw0 * input.y;

    m_weights[0] += dw0;
    m_weights[1] += dw1;
    m_weights[2] += dw2;

    // sum of updates to weights is less then our flatness value then
    // we decalre that no progress is made
    if( dw0 * dw0 + dw1 * dw1 + dw2 * dw2 <= s_flatness )
    {
        return true;
    }

    return false;
}


bool NeuralNetwork::NeuralLayer::Neuron::updateWeights( const std::vector<float> & input )
{
    float dw0,dw1,dw2;
    // w <-- w - theta * x^(l-1)*Delta^l
    // iterate through all weights of this neuron and update its weights
    dw0 = -this->m_delta * s_theta;
    dw1 = dw0 * input[0];
    dw2 = dw0 * input[1];

    m_weights[0] += dw0;
    m_weights[1] += dw1;
    m_weights[2] += dw2;

    // sum of updates to weights is less then our flatness value then
    // we decalre that no progress is made
    if( dw0 * dw0 + dw1 * dw1 + dw2 * dw2 <= s_flatness )
    {
        return true;
    }

    return false;
}
