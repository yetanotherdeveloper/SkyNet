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

/*! The way gradient is computed :
 * STOCHASTIC -- one sample at time
 * BATCH -- all available samples
 */
enum class GradientDescentType {STOCHASTIC, BATCH};
/*! This enum represents the way we initialize weights of neuron
 *  Note: Only INIT_RANDOM is to be ultimately used
 *  INIT_ONE and INIT_ZERO are added just for diagnostic purposes
 *  Having weights iniitialized to any fixed value (0,1 or any other)
 *  increase a danager of falling into local minimum
 */
enum class NeuronFlags {INIT_RANDOM, INIT_ZERO, INIT_ONE};

class NeuralNetwork : public ISkyNetClassificationProtocol
{
    /// Struct representing single layer of Neural Network
    struct NeuralLayer
    {
        /// Struct representing single Neuron
        struct Neuron
        {
private:
            static float s_theta; /// learning grade
            static float s_flatness; /// Value below which sum of weights updates, we declare as flat (no update)
            static const float minRandValue; /// value used to set minimal limit for randomization (TODO: we may derive this from problem description)
            static const float maxRandValue; /// value used to set maximal limit for randomization
            static std::uniform_real_distribution< float > s_randFloat;
            static std::random_device s_rd;
            std::vector< float > m_weights;
            float m_output; /// Recently calculated output
            float m_delta;  /// Recently calculated delta(gradE/gradSignal) for neuron
public:
            Neuron( unsigned int numInputs, NeuronFlags flags = NeuronFlags::INIT_RANDOM ); /// weights initialization
            ~Neuron();
            // There are to functions calculating or getting  output as on first layer we have a points as input , on next layers it is floats given as input
            float getOutput();
            float getOutput(const std::vector<float> & input);
            void setDelta(float deltaValue);
            float getDelta();
            void setWeight( unsigned int index, float value );
            float getWeight(unsigned int index);
            unsigned int getWeightsQuantity();
            void updateWeights( const std::vector<float> & input ); // Update weights of Neuron
        };
public:
        std::vector<Neuron> m_neurons;
        NeuralLayer(unsigned int nrInputs,unsigned int nrNeurons, NeuronFlags flags = NeuronFlags::INIT_RANDOM);
        ~NeuralLayer();
    };


private:
    std::vector<NeuralLayer>            m_layers;  /// Layers of Neural Network
    std::vector<int>                    m_classification;   /// Last results of classification query (getClassification)
    std::string                         m_about;
    GradientDescentType                 m_gradType;
public:
    NeuralNetwork( unsigned int nrInputs, unsigned int nrLayers, GradientDescentType gdtype = GradientDescentType::STOCHASTIC, NeuronFlags flags = NeuronFlags::INIT_RANDOM);
    ~NeuralNetwork();
    const std::vector<float> & RunCL(const std::vector<std::vector<float>> &trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter);

    void RunRef( const std::vector< std::vector<float> > &trainingData,
                 const std::vector<int> &trainingLabels,
                 const std::vector<std::vector<float>>   &validationData,
                 const std::vector<int> &validationLabels,
                 unsigned int max_iterations,
                 SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter);


    void setWeights(std::vector< float > &initial_weights);
    const std::string About() const;
    static std::string composeAboutString();
    float getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels);

    std::vector<int> & getClassification(const std::vector<std::vector<float>> & data);

    void reshape(unsigned int num_inputs, unsigned int num_categories);
    float getNetworkOutput(const std::vector<float> &randomSample);
private:
    void getAllWeights(std::vector< float > &all_weights);
    float getSampleClassificationError(const int sample,float output);
    void updateWeights( const std::vector<float> &randomSample, const int correspondingLabel );
    void updateWeights(const std::vector< std::vector<float> > & trainingData, 
                                                    const std::vector<int> &trainingLabels);
};
#endif //__NN__

