#ifndef __SKYNET__
#define __SKYNET__
#include <string>
#include <vector>
#include <CL/cl.hpp>
#include "protocol.h"

namespace cl
{
    class Context;
    class CommandQueue;
}

class SkyNet
{
    typedef struct
    {
        ISkyNetClassificationProtocol* module;
        void* libHandle;
    }classificationModule;

public:
    SkyNet(void);
    ~SkyNet();
    void RunTests();
private:
    void LoadModules(std::string modulesDirectoryName);
    void InitDevices();
private:
    std::vector<classificationModule> m_classifiers;
    std::vector<cl::Device> m_devices;
};
#endif //__SKYNET__
