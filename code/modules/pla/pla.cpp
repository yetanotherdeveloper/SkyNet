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

// routine to calculate classification and pick missclassiffied point
point& PerceptronLearningAlgorithm::getMisclassifiedPoint(const std::vector<point> & trainingData)
{

}


// TODO: move this constant to some other area or make it derived based on number of training  points
#define MAX_ITERATIONS 100000
void PerceptronLearningAlgorithm::RunRef(const std::vector<point> & trainingData)
{
    // This is where learning takes place
    
    // w(k+1) = w(k) + y(j)x(j)
    
    for(int i=0; i< MAX_ITERATIONS ; ++i) {
       // get random misclassified point 
       // update weight

    } 


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
