#ifndef __SKYNET__
#define __SKYNET__
#include <string>
#include <vector>
#include "protocol.h"

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
private:
    void LoadModules(std::string modulesDirectoryName);
private:
    std::vector<classificationModule> m_classifiers;
};
#endif //__SKYNET__
