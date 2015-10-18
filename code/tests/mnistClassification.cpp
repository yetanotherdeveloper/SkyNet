#include "mnistClassification.h"

mnistClassification::mnistClassification(std::string mnist_dirname)
{

}

namespace {

void runTest(SkyNet& skynet_instance)
{
    std::cout << "MNIST Test executing!" << std::endl;

    auto classifiers = skynet_instance.getClassificationModules();

    for(auto& classifier : classifiers) 
    {
    
    }
}
auto a = TestsRegistry::inst().addTest("MNIST",runTest);
}
