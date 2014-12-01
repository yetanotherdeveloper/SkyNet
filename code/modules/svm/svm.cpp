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


std::vector<int> & SupportVectorMachine::getClassification(const std::vector<point> & data)
{
    m_classification.clear();
    // each data point has corressponding classification info
    // so we can reserve space upfront
    m_classification.reserve(data.size());

    for( unsigned int k = 0; k < data.size(); ++k )
    {
        float result = 1.0f;
            // TODO: implement SVM classification 
        m_classification.push_back(result);
    }
    return m_classification;
}

float SupportVectorMachine::getError(const std::vector<point> & data)
{
    return 1.0f;
}


const std::vector<float> & SupportVectorMachine::RunCL(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic)
{
}


const std::vector<float> & SupportVectorMachine::RunRef(const std::vector<point> & trainingData, SkyNetDiagnostic &diagnostic)
{
    return m_weights;
}


const std::string SupportVectorMachine::About() const
{
    return m_about;
}
