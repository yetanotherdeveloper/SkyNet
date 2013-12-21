#ifndef __PLA__
#define __PLA__
#include <string>
#include "protocol.h"

class SupportVectorMachine : public ISkyNetClassificationProtocol
{
    private:
        std::string m_about;
    public:
        SupportVectorMachine();
        void Run();
        void About();
        const std::string About() const;
};
#endif //__PLA__

