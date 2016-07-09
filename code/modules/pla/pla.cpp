#include "pla.h"
#include <cmath>

extern "C" ISkyNetClassificationProtocol* CreateModule()
{
    return new PerceptronLearningAlgorithm();
}

/*! Build kernels , initialize data
 *
 */
PerceptronLearningAlgorithm::PerceptronLearningAlgorithm() : m_about(PerceptronLearningAlgorithm::composeAboutString() )
{
    m_weights = std::vector<float>(3,0.0f);
}

std::string PerceptronLearningAlgorithm::composeAboutString()
{
    std::string aboutString;
    aboutString.insert(0,"Perceptron Learning Algorithm");
    return aboutString;
}

void PerceptronLearningAlgorithm::reshape(unsigned int num_inputs, unsigned int num_categories)
{
  if(m_weights.size() != num_inputs)
  {
    m_weights.resize(num_inputs,0.0f);
  }
}

std::vector<int> & PerceptronLearningAlgorithm::getClassification(const std::vector<std::vector<float>> & data)
{
    m_classification.clear();
    // each data std::vector<float> has corressponding classification info
    // so we can reserve space upfront
    m_classification.reserve(data.size());

    for( unsigned int k = 0; k < data.size(); ++k )
    {
        m_classification.push_back(classifyPoint(data[k]));
    }
    return m_classification;
}

int PerceptronLearningAlgorithm::classifyPoint(const std::vector<float> &rpoint)
{
    return rpoint[0] * m_weights[1] + m_weights[2] * rpoint[1] + m_weights[0] >= 0.0f ? 1 : -1;
}

void PerceptronLearningAlgorithm::setWeights(std::vector< float > &initial_weights)
{
    for(unsigned int i =0; i< m_weights.size(); ++i) 
    {
        m_weights[i] = initial_weights[i];
    }
}

// routine to calculate classification and pick missclassiffied std::vector<float>
bool PerceptronLearningAlgorithm::getMisclassifiedPoint(const std::vector<std::vector<float>> & trainingData, const std::vector<int> &trainingLabels, const std::vector<float>** outputData, const int** outputLabel)
{

    for(unsigned int i=0; i < trainingData.size(); ++i)
    {

        if( classifyPoint( trainingData[i] ) * trainingLabels[i] < 0)
        {
            *outputData = &(trainingData[i]);
            *outputLabel = &(trainingLabels[i]);
            return true;
        }
    }
    return false;
}

/*! Given sample classification error
 *  Square error is used as error measure eg
 *  (perceptron_value - sample_classification )^2
 */
float PerceptronLearningAlgorithm::getSampleClassificationError( const int sample, float output )
{
    return powf( (output - ( float )sample), 2.0f );
}

float PerceptronLearningAlgorithm::getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels)
{
    // TODO: adjust capacity
    float total_error = 0.0f;

    // Send each std::vector<float> through NN and get classification error for it
    // later on all errors are summed up and divided by number of samples
    for( unsigned int k = 0; k < data.size(); ++k )
    {
        total_error += getSampleClassificationError( labels[k], classifyPoint(data[k]) ); // TODO: this is wrong!!
    }
    return total_error / ( float )data.size();
}

// Update weights according the rule: w_k+1 <-- w_k + y_t*x_t
void PerceptronLearningAlgorithm::updateWeights(const std::vector<float>& rpoint, const int classification)
{
  m_weights[0] = m_weights[0] + (float)classification;
  for(int i = 1; i<m_weights.size(); ++i )
  {
    m_weights[i] = m_weights[i] + (float)classification*rpoint[i-1]; 
  }  
}

void PerceptronLearningAlgorithm::RunRef( const std::vector< std::vector<float> > &trainingData,
                                          const std::vector<int> &trainingLabels,
                                          const std::vector<std::vector<float>>   &validationData,
                                          const std::vector<int> &validationLabels,
                                          unsigned int max_iterations,
                                          SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter)
{
    //const int max_iterations = 1000*trainingData.size();
    const std::vector<float>* misclassifiedData = nullptr;
    const int* misclassifiedLabel = nullptr;
    int i=0;
    bool finish = false;
    diagnostic.storeWeightsAndError(m_weights,getError(trainingData,trainingLabels), getError(validationData,validationLabels) );
    while((i<max_iterations)&&(finish == false))  {
        finish = !getMisclassifiedPoint(trainingData,trainingLabels,&misclassifiedData,&misclassifiedLabel);
        // Check if user want to cease learning
        finish = finish || exitter();
        if(finish == false) {
            updateWeights(*misclassifiedData,*misclassifiedLabel);
            diagnostic.storeWeightsAndError(m_weights,getError(trainingData,trainingLabels), getError(validationData,validationLabels) );
            ++i;
        }
    } 
    if(finish == false) {
        printf("Warning: Perceptron Learning alogorithm exhusted all iterations. This may mean that data is not lineary separable or not enough iterations is allowed!\n");
    }
    return;
}


const std::vector<float> & PerceptronLearningAlgorithm::RunCL(const std::vector<std::vector<float>> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter)
{
    float testValue = 0.0f;
}




const std::string PerceptronLearningAlgorithm::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

PerceptronLearningAlgorithm::~PerceptronLearningAlgorithm()
{
}
