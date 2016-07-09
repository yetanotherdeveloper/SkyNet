#include "nn.h"
#include <random>
#include <cmath>
#include <limits>
#include <cassert>

extern "C" ISkyNetClassificationProtocol *CreateModule()
{
    return new NeuralNetwork( 2, 2, GradientDescentType::STOCHASTIC );  // Just for testing, architecture depends heavily on problem we are to
                                                // solve
}

/*! Build kernels , initialize data
 *
 */
NeuralNetwork::NeuralNetwork( unsigned int nrInputs, unsigned int nrLayers, GradientDescentType gdtype) : 
                              m_about( NeuralNetwork::composeAboutString() ), m_gradType(gdtype)
{
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
/*! Modify number of inputs eg. number of connections to each neuron of first neural layer.
 *  This is needed for neural network to work with diffrent size of input data
 */
void NeuralNetwork::reshape(unsigned int num_inputs, unsigned int num_categories)
{
    // In case requested number of inputs differ
    // from existing setting
    // recreate first neural layer of network with
    // requested number of settings
    const unsigned int nrLayers = m_layers.size();
    const unsigned int current_num_inputs = m_layers[0].m_neurons[0].getWeightsQuantity();

    if(current_num_inputs != num_inputs)
    {
      m_layers[0] = NeuralLayer( num_inputs, 
                                 ( unsigned int )powf( 2.0f,
                                 ( float )(nrLayers - 1) ),
                                 NeuronFlags::INIT_RANDOM );
    }
    // TODO: Reinitialize weights (other layers)
}

std::string NeuralNetwork::composeAboutString()
{
    std::string aboutString;
    aboutString.insert( 0, "Neural Network Algorithm " );
    return aboutString;
}

//TODO: rewrite so it match classification of many categories
std::vector<int> & NeuralNetwork::getClassification(const std::vector<std::vector<float>> & data)
{
    m_classification.clear();
    // each data std::vector<float> has corressponding classification info
    // so we can reserve space upfront
    m_classification.reserve(data.size() );

    // Send each std::vector<float> through NN and get classification error for it
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
float NeuralNetwork::getSampleClassificationError( const int sample, float output )
{
    return powf( (output - ( float )sample), 2.0f );
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
float NeuralNetwork::getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels   )
{
    // TODO: adjust capacity
    float total_error = 0.0f;

    // Send each std::vector<float> through NN and get classification error for it
    // later on all errors are summed up and divided by number of samples
    for( unsigned int k = 0; k < data.size(); ++k )
    {
        total_error += getSampleClassificationError( labels[k], getNetworkOutput(data[k]) );
    }
    return total_error / ( float )data.size();

}


float NeuralNetwork::getNetworkOutput(const std::vector<float> &randomSample)
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
void NeuralNetwork::updateWeights( const std::vector<float> &randomSample, const int correspondingLabel )
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
    m_layers[m_layers.size() - 1].m_neurons[0].setDelta( 2.0f*( output[0] - (float)correspondingLabel )*(1.0f - output[0] * output[0]) );

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

    // update first layer
    for( unsigned int j = 0; j < m_layers[0].m_neurons.size(); ++j )
    {
        m_layers[0].m_neurons[j].updateWeights( randomSample );
    }

    // update hidden layers
    for( unsigned int i = 1; i < m_layers.size(); ++i )
    {
        for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
        {
            // pass as input to neurons of given layer an output of previous layer
            m_layers[i].m_neurons[j].updateWeights( neurons_outputs[i - 1] );
        }
    }
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
void NeuralNetwork::updateWeights(const std::vector< std::vector<float> > & trainingData, 
                                                    const std::vector<int> &trainingLabels)
{
    // Get final delta: de/ds^l
    std::vector< float >             input;
    std::vector< float >             output;
    std::vector<std::vector<float> > neurons_outputs;    // All outputs are here 0 - first layer, 1 - first hidden..

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
        m_layers[m_layers.size() - 1].m_neurons[0].setDelta( 2.0f*( output[0] - (float)trainingLabels[s])*(1.0f - output[0] * output[0]) );

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
            m_layers[0].m_neurons[j].updateWeights( trainingData[s] );
        }

        // update hidden layers
        for( unsigned int i = 1; i < m_layers.size(); ++i )
        {
            for( unsigned int j = 0; j < m_layers[i].m_neurons.size(); ++j )
            {
                // pass as input to neurons of given layer an output of previous layer
                m_layers[i].m_neurons[j].updateWeights( neurons_outputs[i - 1] );
            }
        }
    }   //s
}

/////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::RunRef( const std::vector< std::vector<float> > &trainingData,
                                                    const std::vector<int> &trainingLabels,
                                                    const std::vector<std::vector<float>>   &validationData,
                                                    const std::vector<int> &validationLabels,
                                                    unsigned int max_iterations,
                                                    SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter)
{
    std::uniform_int_distribution< int > sample_index( 0, trainingData.size() - 1 );
    std::random_device rd;

    std::vector<float> all_weights;
    getAllWeights(all_weights);
    diagnostic.storeWeightsAndError(all_weights,getError(trainingData,trainingLabels), getError(validationData,validationLabels) );

    unsigned int interval = 1;        // number of iterations after which testing comes and printing

    const unsigned int min_iterations = 1000; // Find better way of defining minimum iterations to be done
    SkyNetEarlyStop es(min_iterations,max_iterations, 0.4f);

    unsigned int i = 1;
    while( (es.earlyStop(i,
                         all_weights,
                         getError(validationData, validationLabels),
                         getError(trainingData, trainingLabels)) == false) && (exitter() == false) )
    {
        //float err_before = getError(trainingData);
        if(m_gradType == GradientDescentType::STOCHASTIC)
        {
            auto sample_idx = sample_index( rd );
            updateWeights( trainingData[sample_idx], trainingLabels[sample_idx] );
        } else {
            updateWeights( trainingData, trainingLabels );
        }

        getAllWeights(all_weights);
        diagnostic.storeWeightsAndError(all_weights,
                                        getError(trainingData, trainingLabels), 
                                        getError(validationData, validationLabels) );
      if(i%interval == 0) {
        std::cout << "It:" << i << " Train Error: " << getError(trainingData, trainingLabels) << " Val Error: " 
                  << getError(validationData, validationLabels) << std::endl;
      }
      ++i;
    }
    // Get optimal found weights to be final weights
    es.getOptimalWeights(all_weights);
    setWeights(all_weights);
    diagnostic.storeWeightsAndError(all_weights,
                                    getError(trainingData, trainingLabels), 
                                    getError(validationData, validationLabels) );

    return;
}


const std::vector< float > & NeuralNetwork::RunCL( const std::vector< std::vector<float> > &trainingData,
                                                   SkyNetDiagnostic           &diagnostic,
                                                   SkynetTerminalInterface& exitter)
{
    float testValue = 0.0f;
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

unsigned int NeuralNetwork::NeuralLayer::Neuron::getWeightsQuantity()
{
    return m_weights.size();
}

void NeuralNetwork::NeuralLayer::Neuron::updateWeights( const std::vector<float> & input )
{
    float dw0,dw1,dw2;
    // w <-- w - theta * x^(l-1)*Delta^l
    // iterate through all weights of this neuron and update its weights
    dw0 = -this->m_delta * s_theta;
    m_weights[0] += dw0;
    for(unsigned int i = 1; i < m_weights.size(); ++i) {
        m_weights[i] += dw0 * input[i-1];
    }
}
