#ifndef __PLA__
#define __PLA__
#include <string>
#include "protocol.h"

namespace cl
{
    class Device;
    class Context;
    class CommandQueue;
}

class SupportVectorMachine : public ISkyNetClassificationProtocol
{
    private:
        std::string m_about;
        cl::Context *m_context;
        cl::CommandQueue *m_queue;
        const cl::Device *const m_pdevice;
    public:
        SupportVectorMachine(const cl::Device* const pdevice);
        ~SupportVectorMachine();
        void Run();
        void About();
        const std::string About() const;
};
#endif //__PLA__

