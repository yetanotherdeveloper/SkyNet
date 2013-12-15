#ifndef __SKYNET__
#define __SKYNET__
#include <string>
class SkyNet
{
    public:
        SkyNet(void);
        ~SkyNet();
        void LoadModules(std::string modulesDirectoryName);
};
#endif //__SKYNET__
