#include "pla.h"

extern "C" ISkyNetClassificationProtocol* CreateModule()
{
    return new PerceptronLearningAlgorithm();
}

PerceptronLearningAlgorithm::PerceptronLearningAlgorithm() : m_about("Perceptron Learning Algorithm")
{

}

void PerceptronLearningAlgorithm::Run(){
}

const std::string PerceptronLearningAlgorithm::About() const
{
    return m_about;
}
