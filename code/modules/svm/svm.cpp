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
    delete m_context;
    m_context = NULL;
    delete m_queue; 
    m_queue = NULL;
}

void SupportVectorMachine ::Run(){
}
const std::string SupportVectorMachine::About() const
{
    return m_about;
}
