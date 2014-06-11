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
GradientDescent::GradientDescent(const cl::Device* const pdevice) : m_about(GradientDescent::composeAboutString(pdevice) ), m_pdevice(pdevice), m_theta(0.1) , m_flatness(0.0000001f)

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
 *            divided by number of samples 
 *
 */ 
bool GradientDescent::updateWeights( const std::vector< point > & trainingData )
{
    float                                dw0, dw1, dw2, tmpVal;
    std::vector< point >::const_iterator it;

    dw0 = 0.0f;
    dw1 = 0.0f;
    dw2 = 0.0f;
    for( it = trainingData.begin(); it != trainingData.end(); ++it )
    {
        tmpVal = -m_theta * 2.0 * ( it->x * m_weights[1] + m_weights[2] * it->y + m_weights[0] - it->classification );
        dw0   += tmpVal;         // gradient per w0
        dw1   += tmpVal * it->x; // gradient per w1
        dw2   += tmpVal * it->y; // gradient per w2
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


// TODO: move this constant to some other area or make it derived based on number of training  points
const std::vector<float> & GradientDescent::RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,SkyNetDiagnostic &diagnostic)              
{
    m_weights = initial_weights;
    const int max_iterations = 1000*trainingData.size();
    int i=0;
    bool finish = false;
    diagnostic.storeWeights(m_weights);
    while((i<max_iterations)&&(finish == false))  {
        finish = updateWeights(trainingData);
        diagnostic.storeWeights(m_weights);
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


const std::string GradientDescent::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

GradientDescent::~GradientDescent()
{
}