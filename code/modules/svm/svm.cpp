#include "svm.h"

extern "C" ISkyNetClassificationProtocol* CreateModule()
{
    return new SupportVectorMachine();
}


SupportVectorMachine::SupportVectorMachine() : m_about("Support Vector Machine")
{

}

void SupportVectorMachine ::Run(){
}
const std::string SupportVectorMachine::About() const
{
    return m_about;
}
