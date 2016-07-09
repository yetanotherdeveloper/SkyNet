#include "gd.h"

extern "C" ISkyNetClassificationProtocol* CreateModule()
{
    return new GradientDescent();
}

/*! Build kernels , initialize data
 *
 */
GradientDescent::GradientDescent() : m_about(GradientDescent::composeAboutString() ), m_theta(0.1) , m_flatness(0.0000001f)

{
    m_weights = std::vector<float>(3,0.0f);
}

void GradientDescent::reshape(unsigned int num_inputs, unsigned int num_categories)
{
  if(m_weights.size() != num_inputs)
  {
    m_weights.resize(num_inputs,0.0f);
  }
}

std::string GradientDescent::composeAboutString()
{
    std::string aboutString;
    aboutString.insert(0,"Gradient Descent Algorithm");
    return aboutString;
}


std::vector<int> & GradientDescent::getClassification(const std::vector<std::vector<float>> & data)
{
    m_classification.clear();
    // each data point has corressponding classification info
    // so we can reserve space upfront
    m_classification.reserve(data.size());

    for( unsigned int k = 0; k < data.size(); ++k )
    {
        float result = 1.0f;

        for(unsigned int i = 0; i < m_weights.size(); i += 3)
        {
            result *= (m_weights[i+1] * data[k][0] + m_weights[i+2] * data[k][1] + m_weights[i]);
        }
        m_classification.push_back(result > 0.0f ? 1 : -1);
    }
    return m_classification;
}


// TODO: make it working for any dimentions not just two
/*! Function updating weights based on current gradient descent
 *  Desc: 
 *          Update rule: w_k+1 <-- w_k - theta*grad(E_in(w(k))          
 *          E_in(w(t)) = (x*w(t) - target_value )**2
 *          dE_in/dw(t) = 2*(x*w(t) - target_value)
 *
 *          where:
 *            x*w(t) is learned_value
 *            E_in(w(t)) is square error eg. sum of square errors over all training set 
 *            divided by number of samples 
 *
 */ 
bool GradientDescent::updateWeights( const std::vector< std::vector<float> > & trainingData, const std::vector<int>& classificationData )
{
    float                                dw0, dw1, dw2, tmpVal;
    std::vector< std::vector<float> >::const_iterator it;

    dw0 = 0.0f;
    dw1 = 0.0f;
    dw2 = 0.0f;
    for( int i=0; i< trainingData.size(); ++i )
    {
        tmpVal = -m_theta * 2.0 * ( trainingData[i][0] * m_weights[1] + m_weights[2] * trainingData[i][1] + m_weights[0] - classificationData[i] );
        dw0   += tmpVal;         // gradient per w0
        dw1   += tmpVal * trainingData[i][0]; // gradient per w1
        dw2   += tmpVal * trainingData[i][1]; // gradient per w2
    }

    m_weights[0] += dw0/trainingData.size();
    m_weights[1] += dw1/trainingData.size();
    m_weights[2] += dw2/trainingData.size();

    // sum of updates to weights is less then our flatness value then
    // we decalre that no progress is made
    if( dw0 * dw0 + dw1 * dw1 + dw2 * dw2 <= m_flatness )
    {
        return true;
    }

    return false;
}
////////////////////////////////////////////////////////////////////////////
void GradientDescent::setWeights(std::vector< float > &initial_weights)
{
    for(unsigned int i =0; i< m_weights.size(); ++i) 
    {
        m_weights[i] = initial_weights[i];
    }
}
////////////////////////////////////////////////////////////////////////////
float GradientDescent::getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels)
{
    //TODO: implement it
    return 1.0f;
}


void GradientDescent::RunRef( const std::vector< std::vector<float> > &trainingData,
                              const std::vector<int> &trainingLabels,
                              const std::vector<std::vector<float>>   &validationData,
                              const std::vector<int> &validationLabels,
                              unsigned int max_iterations,
                              SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter)
{
    int i=0;
    bool finish = false;
    diagnostic.storeWeightsAndError(m_weights,0.0f,0.0f);
    while((i<max_iterations)&&(finish == false))  {
        finish = updateWeights(trainingData,trainingLabels);
        //TODO: Store proper error
        diagnostic.storeWeightsAndError(m_weights,0.0f,0.0f);
        finish = finish || exitter();
        if(finish == false) {
            ++i;
        }
    } 
    if(finish == false) {
        printf("Warning: Gradient Descent Learning alogorithm exhusted all iterations. TODO: Make proper termination criteria\n");
    }
    return;
}

const std::vector<float> & GradientDescent::RunCL(const std::vector<std::vector<float>> &trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter)
{
    float testValue = 0.0f;
}


const std::string GradientDescent::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

GradientDescent::~GradientDescent()
{
}
