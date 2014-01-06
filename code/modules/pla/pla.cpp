#include "pla.h"
#include <CL/cl.hpp>

extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)
{
    return new PerceptronLearningAlgorithm(pdevice);
}

/*! Build kernels , initialize data
 *  
*/
PerceptronLearningAlgorithm::PerceptronLearningAlgorithm(const cl::Device* const pdevice) : m_about("Perceptron Learning Algorithm"), m_pdevice(pdevice)
{
    cl_int err;

   // TODO:
   // - create command queue
   // - build programs, make kernels
   // - store device, command queue, context


    // make a context , command queue
    m_context = new cl::Context(*m_pdevice,NULL,NULL,NULL,&err); 

    if( err != CL_SUCCESS ) {
        //SKYNET_INFO(" Error creating OpenCL context: %d\n",err);
        printf("Error creating OpenCL context for PLA module: %d\n",err);
    }
    
    std::vector<cl::Device> context_devices;
    m_context->getInfo(CL_CONTEXT_DEVICES,&context_devices);

    m_queue = new cl::CommandQueue(*m_context,*m_pdevice,0,&err); 

    if( err != CL_SUCCESS ) {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error creating OpenCL command queue: %d\n",err);
    }
}

void PerceptronLearningAlgorithm::Run(){
}

const std::string PerceptronLearningAlgorithm::About() const
{
    return m_about;
}

PerceptronLearningAlgorithm::~PerceptronLearningAlgorithm()
{
    delete m_context;
    m_context = NULL;
    delete m_queue; 
    m_queue = NULL;
}
