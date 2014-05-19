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
        std::unique_ptr<cl::Kernel> m_plaKernel;
        const cl::Device *const m_pdevice;
    public:
        PerceptronLearningAlgorithm(const cl::Device* const pdevice);
        ~PerceptronLearningAlgorithm();
        void RunCL();
        void RunRef(const std::vector<point> & trainingData);              
        const std::string About() const;
        static std::string composeAboutString(const cl::Device* const pdevice);
    private:
        int classifyPoint(const point &rpoint, float &w0, float &w1, float &w2);
        bool getMisclassifiedPoint(const std::vector<point> & trainingData, float &w0, float &w1, float &w2, point* output);
};
#endif //__PLA__

