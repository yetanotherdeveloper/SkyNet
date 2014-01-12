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
PerceptronLearningAlgorithm::PerceptronLearningAlgorithm(const cl::Device* const pdevice) : m_about("Perceptron Learning Algorithm"), m_pdevice(pdevice)
{
    cl_int err;

    // TODO:
    // - create command queue
    // - build programs, make kernels
    // - store device, command queue, context


    // make a context , command queue
    m_context = new cl::Context(*m_pdevice,NULL,NULL,NULL,&err);

    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL context: %d\n",err);
        printf("Error creating OpenCL context for PLA module: %d\n",err);
    }

    std::vector<cl::Device> context_devices;
    m_context->getInfo(CL_CONTEXT_DEVICES,&context_devices);

    m_queue = new cl::CommandQueue(*m_context,*m_pdevice,0,&err);

    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error creating OpenCL command queue: %d\n",err);
    }

    // Create/build program, get kernels

    cl::Program::Sources kern_sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length() + 1));
    cl::Program program(*m_context, kern_sources,&err);
    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error creating OpenCL program from source: %d\n",err);
    }

    std::vector<cl::Device> targetDevices(1,*m_pdevice);
    err = program.build(targetDevices, "-cl-std=CL1.0");
    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error Building OpenCL program failed: %d\n",err);
        std::string log; 
        err =  program.getBuildInfo(*m_pdevice,CL_PROGRAM_BUILD_LOG,&log);
        printf(" OpenCL program build log: %s\n",log.c_str());
    }

    m_plaKernel = new cl::Kernel(program, "dodaj", &err);
    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error creating OpenCL PLA kernel : %d\n",err);
    }


    //TODO: error handling section

}

void PerceptronLearningAlgorithm::Run()
{
    float testValue = 0.0f;
    m_plaKernel->setArg(0,&testValue);
    m_queue->enqueueTask(*m_plaKernel);
}

const std::string PerceptronLearningAlgorithm::About() const
{
    // TODO: Return Also Device we are running for
    return m_about;
}

PerceptronLearningAlgorithm::~PerceptronLearningAlgorithm()
{
    delete m_plaKernel;
    m_plaKernel = NULL;
    delete m_context;
    m_context = NULL;
    delete m_queue;
    m_queue = NULL;
}
