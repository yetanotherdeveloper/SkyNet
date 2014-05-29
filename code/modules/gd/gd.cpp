#include "gd.h"
#include <CL/cl.hpp>

static std::string kernelSource = "__kernel void dodaj(float veciu) \
                                   {  \
                                            veciu = 1.0f; \
                                   }";

extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)
{
    return new GradientDescent(pdevice);
}

/*! Build kernels , initialize data
 *
 */
GradientDescent::GradientDescent(const cl::Device* const pdevice) : m_about(GradientDescent::composeAboutString(pdevice) ), m_pdevice(pdevice), m_theta(0.1) 
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

    m_diagnostic =  std::unique_ptr<SkyNetDiagnostic>(new SkyNetDiagnostic(GradientDescent::composeAboutString(pdevice)));
    //TODO: error handling section

}


std::string GradientDescent::composeAboutString(const cl::Device* const pdevice)
{
    std::string aboutString;
    pdevice->getInfo(CL_DEVICE_NAME, &aboutString);
    aboutString.insert(0,"Gradient Descent Algorithm (");
    aboutString.append(")");
    return aboutString;
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
 *
 */ 
bool GradientDescent::updateWeights(const std::vector<point> & trainingData)
{
    std::vector<point>::const_iterator it;
    for(it = trainingData.begin(); it != trainingData.end(); ++it)
    {
        m_weights[0] += -m_theta*2.0*(it->x * m_weights[1] + m_weights[2] * it->y + m_weights[0] - it->classification);       // gradient per w0
        m_weights[1] += -m_theta*2.0*(it->x * m_weights[1] + m_weights[2] * it->y + m_weights[0] - it->classification)*it->x; // gradient per w1
        m_weights[2] += -m_theta*2.0*(it->x * m_weights[1] + m_weights[2] * it->y + m_weights[0] - it->classification)*it->y; // gradient per w2
    }
    return false;
}


// TODO: move this constant to some other area or make it derived based on number of training  points
const std::vector<float> & GradientDescent::RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights)              
{
    m_weights = initial_weights;
    const int max_iterations = 1000*trainingData.size();
    int i=0;
    bool finish = false;
    m_diagnostic->storeWeights(m_weights);
    while((i<max_iterations)&&(finish == false))  {
        finish = updateWeights(trainingData);
        m_diagnostic->storeWeights(m_weights);
        if(finish == false) {
            ++i;
        }
    } 
    if(finish == false) {
        printf("Warning: Gradient Descent Learning alogorithm exhusted all iterations. TODO: Make proper termination criteria\n");
    }
    return m_weights;
}


void GradientDescent::RunCL()
{
    float testValue = 0.0f;
    m_plaKernel->setArg(0,&testValue);
    m_pCommandQueue->enqueueTask(*m_plaKernel);
}


bool GradientDescent::makeDiagnostic()
{
    m_diagnostic->dumpWeights();
}


const std::string GradientDescent::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

GradientDescent::~GradientDescent()
{
}
