#include <dlfcn.h>
#include "os_inc.h"
#include "skynet.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>


typedef ISkyNetClassificationProtocol* (*pCreateModule)();

/* Linux specific code of loading potential module
 *
 * return: pointer on created module, NULL if loading failed
 */
ISkyNetClassificationProtocol* SkyNetOS::LoadModule(std::string moduleName, void** plibHandle)
{
    *plibHandle = NULL;
    pCreateModule pCM = NULL;

    // Name need to consists of ".so"
    if( moduleName.find(".so") == std::string::npos )
    {
        return NULL;
    }

    // Load Library
    *plibHandle = dlopen(moduleName.c_str(),RTLD_NOW);
    if(*plibHandle == NULL)
    {
        SKYNET_DEBUG("Loading of module: %s failed due to: %s\n",moduleName.c_str(),dlerror() );
        return NULL;
    }

    // Get CreateModule
    pCM = (pCreateModule)dlsym(*plibHandle,"CreateModule");
    if(pCM == NULL)
    {
        SKYNET_DEBUG("Loading of module: %s failed due to: %s\n",moduleName.c_str(),dlerror() );
        return NULL;
    }

    // call CreateModule
    return pCM();
}

void SkyNetOS::ReleaseModule(ISkyNetClassificationProtocol** pModule,void** pLibHandle)
{
    delete *pModule;
    *pModule = NULL;
    dlclose(*pLibHandle);
    *pLibHandle = NULL;
}


unsigned int SkyNetOS::getPID()
{
    return getpid();
}

bool SkyNetOS::CreateDirectory(const std::string& dirname)
{
    //TODO: Make recursive directory creating
    std::size_t found = dirname.find_first_of("/",1);   //Find second slash

    //std::string::npos

    if(mkdir(dirname.c_str(),0777) == 0)
    {
        SKYNET_DEBUG("Directory: %s created\n",dirname.c_str() );
        return true;
    } else {
        // If directory already exists then it is fine as well!
        if(errno == EEXIST) {
            SKYNET_DEBUG("Directory: %s already exists!\n",dirname.c_str() );
            return true;
        }
    }
    perror("Error creating directory: ");
    return false;
}


std::string SkyNetOS::GetHomeDirectory(void)
{
    struct passwd *pw = getpwuid(getuid());
    return std::string(pw->pw_dir);
}

