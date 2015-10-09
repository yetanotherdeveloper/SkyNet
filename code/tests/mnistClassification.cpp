#include "mnistClassification.h"

mnistClassification::mnistClassification(std::string mnist_dirname)
{

}

namespace {

void runTest(void)
{
    std::cout << "MNIST Test executing!" << std::endl;
}
auto a = TestsRegistry::inst().addTest("MNIST",runTest);
}
