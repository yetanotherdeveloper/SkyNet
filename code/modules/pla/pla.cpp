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


int PerceptronLearningAlgorithm::classifyPoint(const point &rpoint, float &w0, float &w1, float &w2)
{
    return rpoint.x * w1 + w2 * rpoint.y + w0 >= 0.0f ? 1 : -1;
}


// routine to calculate classification and pick missclassiffied point
bool PerceptronLearningAlgorithm::getMisclassifiedPoint(const std::vector<point> & trainingData, float &w0, float &w1, float &w2, point* output)
{
    std::vector<point>::iterator it;
    for(it = trainingData.begin(); it != trainingData.end(); ++it) {

        if( classifyPoint( (*it),w0,w1,w2) * it->classification < 0)
        {
            output = &(*it);
            return true;
        }
    }
    return false;
}


// TODO: move this constant to some other area or make it derived based on number of training  points
void PerceptronLearningAlgorithm::RunRef(const std::vector<point> & trainingData, float &w0, float &w1, float &w2)
{
    // w(k+1) = w(k) + y(j)x(j)
    const int max_iterations = 1000*trainingData.size();
    point misclassified;
    int i=0;
    bool finish = false;
    while((i<MAX_ITERATIONS)&&(finish == false))  {
        finish = !getMisclassifiedPoint(trainingData,w0,w1,w2,&misclassified);
        if(finish == false) {
            // update weights        
            ++i;
        }
    } 
    if(finish == false) {
        printf("Warning: Perceptron Learning alogorithm exhusted all iterations. This may mean that data is not lineary separable or not enough iterations is allowed!\n");
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
