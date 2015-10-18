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
public:
    typedef struct
    {
        ISkyNetClassificationProtocol* module;
        void* libHandle;
    }classificationModule;

public:
    SkyNet(int argc, char *const *argv);
    ~SkyNet();
    void RunTests();

    const std::vector<classificationModule>& getClassificationModules(void);    

private:
    void LoadModules(std::string modulesDirectoryName);
    void InitDevices();
    void ProcessCommandLine(int argc, char *const *argv);
    std::string getModuleToLoad(char *fileToLoad);
    void PrintHelp();
    void PrintModules();
    void PrintTests();
private:
    std::vector<classificationModule> m_classifiers;
    std::vector<cl::Device> m_devices;
    bool m_terminated;  //< No further work to be done. Just finish gracefully
    bool m_printmodules;//< Whether to print found modules
    bool m_printTests; //<Whether to print available tests
    unsigned short m_enableModule; 
    unsigned short m_testToExecute; 
    std::string m_moduleToLoad;
    std::vector< float > m_moduleWeights;   //< serialized Weights for given module (to be used for initialization of learning for this module)

};
#endif //__SKYNET__
