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
        std::vector<float> m_weights;
    public:
        SupportVectorMachine(const cl::Device* const pdevice);
        ~SupportVectorMachine();
        void RunCL();
        const std::vector<float> & RunRef(const std::vector<point> & trainingData, const std::vector<float> & initial_weights,
                                         SkyNetDiagnostic &diagnostic);              
        void About();
        const std::string About() const;
};
#endif //__PLA__

