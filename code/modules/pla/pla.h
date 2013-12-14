#ifndef __PLA__
#define __PLA__
#include "protocol.h"

class PerceptronLearningAlgorithm : public ISkyNetClassificationProtocol
{
    public:
        void Run();
        void About();
};
#endif //__PLA__

