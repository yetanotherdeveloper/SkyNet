#include "pla.h"

extern "C" ISkyNetClassificationProtocol* CreateModule()
{
    return new PerceptronLearningAlgorithm();
}


void PerceptronLearningAlgorithm::Run(){
}
void PerceptronLearningAlgorithm::About(){
}
