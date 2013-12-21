#include "svm.h"

extern "C" ISkyNetClassificationProtocol* CreateModule()
{
    return new SupportVectorMachine();
}



void SupportVectorMachine ::Run(){
}
void SupportVectorMachine ::About(){
}
