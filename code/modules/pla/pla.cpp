#include "pla.h"
#include <CL/cl.hpp>
#include <cmath>

static std::string kernelSource = "__kernel void dodaj(float veciu) \
                                   {  \
                                            veciu = 1.0f; \
                                   }";

extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)
{
    return new PerceptronLearningAlgorithm(pdevice);
}

/*! Build kernels , initialize data
 *
 */
PerceptronLearningAlgorithm::PerceptronLearningAlgorithm(const cl::Device* const pdevice) : m_about(PerceptronLearningAlgorithm::composeAboutString(pdevice) ), m_pdevice(pdevice) 
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

    m_weights = std::vector<float>(3,0.0f);
}

std::string PerceptronLearningAlgorithm::composeAboutString(const cl::Device* const pdevice)
{
    std::string aboutString;
    pdevice->getInfo(CL_DEVICE_NAME, &aboutString);
    aboutString.insert(0,"Perceptron Learning Algorithm (");
    aboutString.append(")");
    return aboutString;
}


std::vector<int> & PerceptronLearningAlgorithm::getClassification(const std::vector<point> & data)
{
    m_classification.clear();
    // each data point has corressponding classification info
    // so we can reserve space upfront
    m_classification.reserve(data.size());

    for( unsigned int k = 0; k < data.size(); ++k )
    {
        m_classification.push_back(classifyPoint(data[k]));
    }
    return m_classification;
}


int PerceptronLearningAlgorithm::classifyPoint(const point &rpoint)
{
    return rpoint.x * m_weights[1] + m_weights[2] * rpoint.y + m_weights[0] >= 0.0f ? 1 : -1;
}

void PerceptronLearningAlgorithm::setWeights(std::vector< float > &initial_weights)
{
    for(unsigned int i =0; i< m_weights.size(); ++i) 
    {
        m_weights[i] = initial_weights[i];
    }
}

// routine to calculate classification and pick missclassiffied point
bool PerceptronLearningAlgorithm::getMisclassifiedPoint(const std::vector<point> & trainingData, const point** output)
{
    std::vector<point>::const_iterator it;
    for(it = trainingData.begin(); it != trainingData.end(); ++it) {

        if( classifyPoint(*it) * it->classification < 0)
        {
            *output = &(*it);
            return true;
        }
    }
    return false;
}

/*! Given sample classification error
 *  Square error is used as error measure eg
 *  (perceptron_value - sample_classification )^2
 */
float PerceptronLearningAlgorithm::getSampleClassificationError( const point& sample, float output )
{
    return powf( (output - ( float )sample.classification), 2.0f );
}

float PerceptronLearningAlgorithm::getError(const std::vector<point> & data)
{
    // TODO: adjust capacity
    float total_error = 0.0f;

    // Send each point through NN and get classification error for it
    // later on all errors are summed up and divided by number of samples
    for( unsigned int k = 0; k < data.size(); ++k )
    {
        total_error += getSampleClassificationError( data[k], classifyPoint(data[k]) );
    }
    return total_error / ( float )data.size();
}


// Update weights according the rule: w_k+1 <-- w_k + y_t*x_t
void PerceptronLearningAlgorithm::updateWeights(const point& rpoint)
{
   m_weights[0] = m_weights[0] + (float)rpoint.classification; 
   m_weights[1] = m_weights[1] + (float)rpoint.classification*rpoint.x; 
   m_weights[2] = m_weights[2] + (float)rpoint.classification*rpoint.y; 
}

// TODO: move this constant to some other area or make it derived based on number of training  points
const std::vector<float> & PerceptronLearningAlgorithm::RunRef(const std::vector<point> & trainingData, const std::vector<point> &validationData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter)              
{
    const int max_iterations = 1000*trainingData.size();
    const point* misclassified = nullptr;
    int i=0;
    bool finish = false;
    diagnostic.storeWeightsAndError(m_weights,getError(trainingData), getError(validationData) );
    while((i<max_iterations)&&(finish == false))  {
        finish = !getMisclassifiedPoint(trainingData,&misclassified);
        // Check if user want to cease learning
        finish = finish || exitter();
        if(finish == false) {
            updateWeights(*misclassified);
            diagnostic.storeWeightsAndError(m_weights,getError(trainingData), getError(validationData) );
            ++i;
        }
    } 
    if(finish == false) {
        printf("Warning: Perceptron Learning alogorithm exhusted all iterations. This may mean that data is not lineary separable or not enough iterations is allowed!\n");
    }
    return m_weights;
}


const std::vector<float> & PerceptronLearningAlgorithm::RunCL(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter)
{
    float testValue = 0.0f;
    m_plaKernel->setArg(0,&testValue);
    m_pCommandQueue->enqueueTask(*m_plaKernel);
}




const std::string PerceptronLearningAlgorithm::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

PerceptronLearningAlgorithm::~PerceptronLearningAlgorithm()
{
}
