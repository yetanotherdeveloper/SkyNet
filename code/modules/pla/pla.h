#ifndef __PLA__
#define __PLA__
#include <string>
#include "protocol.h"

namespace cl
{
    class Device;
    class Context;
    class CommandQueue;
    class Kernel;
}

class PerceptronLearningAlgorithm : public ISkyNetClassificationProtocol
{
    private:
        std::string m_about;
        std::unique_ptr<cl::Context> m_pContext;
        std::unique_ptr<cl::CommandQueue> m_pCommandQueue; 
        cl::Kernel* m_plaKernel;
        const cl::Device *const m_pdevice;
    public:
        PerceptronLearningAlgorithm(const cl::Device* const pdevice);
        ~PerceptronLearningAlgorithm();
        void Run();
        const std::string About() const;
        static std::string composeAboutString(const cl::Device* const pdevice);
};
#endif //__PLA__

