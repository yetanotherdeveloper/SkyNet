#include <dlfcn.h>
#include "os_inc.h"
#include "skynet.h"


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
    if( moduleName.find(".so") == std::string::npos ) {
        return NULL;
    }
   
    // Load Library
    *plibHandle = dlopen(moduleName.c_str(),RTLD_NOW);
    if(*plibHandle == NULL) {
        SKYNET_DEBUG("Loading of module: %s failed due to: %s\n",moduleName.c_str(),dlerror());
        return NULL;
    }

    // Get CreateModule
    pCM =  (pCreateModule)dlsym(*plibHandle,"CreateModule");    
    if(pCM == NULL) {
        SKYNET_DEBUG("Loading of module: %s failed due to: %s\n",moduleName.c_str(),dlerror());
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
