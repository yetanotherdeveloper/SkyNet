#include "sgd.h"
#include <CL/cl.hpp>
#include <random>

static std::string kernelSource = "__kernel void dodaj(float veciu) \
                                   {  \
                                            veciu = 1.0f; \
                                   }";

extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)
{
    return new StochasticGradientDescent(pdevice);
}

/*! Build kernels , initialize data
 *
 */
StochasticGradientDescent::StochasticGradientDescent(const cl::Device* const pdevice) : m_about(StochasticGradientDescent::composeAboutString(pdevice) ), m_pdevice(pdevice),
                                                     m_theta(0.1), m_flatness(0.0000001f)
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


std::string StochasticGradientDescent::composeAboutString(const cl::Device* const pdevice)
{
    std::string aboutString;
    pdevice->getInfo(CL_DEVICE_NAME, &aboutString);
    aboutString.insert(0,"Stochastic Gradient Descent Algorithm (");
    aboutString.append(")");
    return aboutString;
}


float StochasticGradientDescent::getError(const std::vector<point> & data)
{
    return 1.0f;
}
////////////////////////////////////////////////////////////////////////////
void StochasticGradientDescent::setWeights(std::vector< float > &initial_weights)
{
    for(unsigned int i =0; i< m_weights.size(); ++i) 
    {
        m_weights[i] = initial_weights[i];
    }
}
////////////////////////////////////////////////////////////////////////////
std::vector<int> & StochasticGradientDescent::getClassification(const std::vector<point> & data)
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
            result *= (m_weights[i+1] * data[k].x + m_weights[i+2] * data[k].y + m_weights[i]);
        }
        m_classification.push_back(result > 0.0f ? 1 : -1);
    }
    return m_classification;
}


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
bool StochasticGradientDescent::updateWeights(const point &randomSample)
{
    // TODO: make it working for any dimentions not just two
    // Perform gradient descent on given (assuming random) sample
    float dw0,dw1,dw2;
    dw0 = -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification);                // gradient per w0
    dw1 = dw0*randomSample.x; // gradient per w1
    dw2 = dw0*randomSample.y; // gradient per w2

    m_weights[0] += dw0;
    m_weights[1] += dw1;
    m_weights[2] += dw2;

    // sum of updates to weights is less then our flatness value then
    // we decalre that no progress is made
    if(dw0*dw0 + dw1*dw1 + dw2*dw2 <= m_flatness) {
        return true;
    }

    //m_weights[0] += -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification);                // gradient per w0
    //m_weights[1] += -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification)*randomSample.x; // gradient per w1
    //m_weights[2] += -m_theta*2.0*(randomSample.x * m_weights[1] + m_weights[2] * randomSample.y + m_weights[0] - randomSample.classification)*randomSample.y; // gradient per w2
    return false;
}


// TODO: move this constant to some other area or make it derived based on number of training  points
const std::vector<float> & StochasticGradientDescent::RunRef(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter)              
{
    std::uniform_int_distribution< int > sample_index( 0, trainingData.size() -1 );
    std::random_device rd;

    const int max_iterations = 1000*trainingData.size();
    int i=0;
    bool finish = false;
    //TODO: Store proper error
    diagnostic.storeWeightsAndError(m_weights,0.0f);
    while((i<max_iterations)&&(finish == false))  {
        finish = updateWeights(trainingData[sample_index(rd)]);
        //TODO: Store proper error
        diagnostic.storeWeightsAndError(m_weights,0.0f);
        // Check if user want to cease learning
        finish = finish || exitter();
        if(finish == false) {
            ++i;
        } else {
            // TODO: If flatness was reached then
            // check if error is below certain error threshold 
        }

    } 
    if(finish == false) {
        printf("Warning: Stochastic Gradient Descent Learning alogorithm exhusted all iterations. TODO: Make proper termination criteria\n");
    }
    return m_weights;
}


const std::vector<float> & StochasticGradientDescent::RunCL(const std::vector<point> &trainingData, SkyNetDiagnostic &diagnostic, SkynetTerminalInterface& exitter)
{
    float testValue = 0.0f;
    m_plaKernel->setArg(0,&testValue);
    m_pCommandQueue->enqueueTask(*m_plaKernel);
}


const std::string StochasticGradientDescent::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

StochasticGradientDescent::~StochasticGradientDescent()
{
}
