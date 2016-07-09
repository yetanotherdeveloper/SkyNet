#ifndef __SKYNET__
#define __SKYNET__
#include <string>
#include <vector>
#include "protocol.h"

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
    const std::string& getMnistDataDir(void);
    const unsigned int getMaxIterations(void);

private:
    void LoadModules(std::string modulesDirectoryName);
    void ProcessCommandLine(int argc, char *const *argv);
    std::string getModuleToLoad(char *fileToLoad);
    void PrintHelp();
    void PrintModules();
    void PrintTests();
private:
    std::vector<classificationModule> m_classifiers;
    bool m_terminated;  //< No further work to be done. Just finish gracefully
    bool m_printmodules;//< Whether to print found modules
    bool m_printTests; //<Whether to print available tests
    unsigned int m_max_iterations; //< Maximum iterations to be executed by ML algorithm
    unsigned short m_enableModule; 
    unsigned short m_testToExecute; 
    std::string m_moduleToLoad;
    std::string m_mnist_dir;        //< Directory holding MNIST data
    std::vector< float > m_moduleWeights;   //< serialized Weights for given module (to be used for initialization of learning for this module)

};
#endif //__SKYNET__
