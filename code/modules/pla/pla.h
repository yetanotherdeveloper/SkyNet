#ifndef __PLA__
#define __PLA__
#include <string>
#include "protocol.h"


class PerceptronLearningAlgorithm : public ISkyNetClassificationProtocol
{
    private:
        std::string m_about;
    public:
        PerceptronLearningAlgorithm();
        void Run();
        const std::string About() const;
};
#endif //__PLA__

