#include <CL/cl.hpp>
#include "svm.h"

extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)
{
    return new SupportVectorMachine(pdevice);
}


SupportVectorMachine::SupportVectorMachine(const cl::Device* const pdevice) : m_about("Support Vector Machine"), m_pdevice(pdevice)
{

}
SupportVectorMachine::~SupportVectorMachine()
{
    //delete m_context;
    //m_context = NULL;
    //delete m_queue;
    //m_queue = NULL;
}


void SupportVectorMachine::RunCL()
{
}


const std::vector<float> & SupportVectorMachine::RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights)
{
    return m_weights;
}

bool SupportVectorMachine::makeDiagnostic()
{
    return true;
}

const std::string SupportVectorMachine::About() const
{
    return m_about;
}
