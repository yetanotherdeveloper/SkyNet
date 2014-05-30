#include "pla.h"
#include <CL/cl.hpp>

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

}

std::string PerceptronLearningAlgorithm::composeAboutString(const cl::Device* const pdevice)
{
    std::string aboutString;
    pdevice->getInfo(CL_DEVICE_NAME, &aboutString);
    aboutString.insert(0,"Perceptron Learning Algorithm (");
    aboutString.append(")");
    return aboutString;
}


int PerceptronLearningAlgorithm::classifyPoint(const point &rpoint)
{
    return rpoint.x * m_weights[1] + m_weights[2] * rpoint.y + m_weights[0] >= 0.0f ? 1 : -1;
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


// Update weights according the rule: w_k+1 <-- w_k + y_t*x_t
void PerceptronLearningAlgorithm::updateWeights(const point& rpoint)
{
   m_weights[0] = m_weights[0] + (float)rpoint.classification; 
   m_weights[1] = m_weights[1] + (float)rpoint.classification*rpoint.x; 
   m_weights[2] = m_weights[2] + (float)rpoint.classification*rpoint.y; 
}

// TODO: move this constant to some other area or make it derived based on number of training  points
const std::vector<float> & PerceptronLearningAlgorithm::RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,
                                                               SkyNetDiagnostic &diagnostic)              
{
    m_weights = initial_weights;
    const int max_iterations = 1000*trainingData.size();
    const point* misclassified = nullptr;
    int i=0;
    bool finish = false;
    diagnostic.storeWeights(m_weights);
    while((i<max_iterations)&&(finish == false))  {
        finish = !getMisclassifiedPoint(trainingData,&misclassified);
        if(finish == false) {
            updateWeights(*misclassified);
            diagnostic.storeWeights(m_weights);
            ++i;
        }
    } 
    if(finish == false) {
        printf("Warning: Perceptron Learning alogorithm exhusted all iterations. This may mean that data is not lineary separable or not enough iterations is allowed!\n");
    }
    return m_weights;
}


void PerceptronLearningAlgorithm::RunCL()
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
