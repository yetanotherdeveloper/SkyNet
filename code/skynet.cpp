#include <dirent.h>
#include <errno.h>
#include <stack>
#include <CL/cl.hpp>

#include "skynet.h"
#include "os_inc.h"


SkyNet::SkyNet(void)
{
    SKYNET_INFO("Skynet Initializing...\n\n");


    SKYNET_INFO("Initializing computing devices:\n");
    InitDevices();


    SKYNET_INFO("Loading Modules:\n");
    LoadModules(std::string("./modules") );
}

SkyNet::~SkyNet()
{
    std::vector<classificationModule>::iterator it;
    //TODO: release all modules and its libs handles
    for(it = m_classifiers.begin(); it != m_classifiers.end(); ++it)
    {
        SkyNetOS::ReleaseModule(&(it->module),&(it->libHandle) );
    }
}

// Scan available computing devices
//
//
void SkyNet::InitDevices()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::string platform_param_value;    
    cl_device_id device;

    if( cl::Platform::get(&platforms) != CL_SUCCESS) {
        SKYNET_INFO("ERROR: No computing platforms found!\n");
        return;
    }

    // List platforms capabilities

    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it ) {

        plat_it->getInfo(CL_PLATFORM_NAME, &platform_param_value);
        SKYNET_INFO("Platform: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_VENDOR, &platform_param_value);
        SKYNET_INFO("   vendor: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_VERSION, &platform_param_value);
        SKYNET_INFO("   version: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_PROFILE, &platform_param_value);
        SKYNET_INFO("   profile: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_EXTENSIONS, &platform_param_value);
        SKYNET_INFO("   extensions: %s\n",platform_param_value.c_str());

        // Now for given platform get list of devices and their capabilities

        plat_id->getDevices();

    }




}

/* Scan directory with modules and loadem all */
void SkyNet::LoadModules(std::string modulesDirectoryName)
{
    // search for files with extension snm
    DIR                          *modules_dir;
    dirent                       * entry;
    std::string                  entryName;
    ISkyNetClassificationProtocol* module;
    void                         * libHandle;
    std::string                  id;
    std::string                  info;


    modules_dir = opendir(modulesDirectoryName.c_str() );
    if(modules_dir != NULL) {
        while( (entry = readdir(modules_dir) ) != NULL) {
            SKYNET_DEBUG("Potential module: %s\n",entry->d_name);
            switch(entry->d_type) {
            // regular file, check if this is DSO and load
            case DT_REG:
                entryName = entry->d_name;
                module    = SkyNetOS::LoadModule(modulesDirectoryName + std::string("/") + entryName,&libHandle);
                if(module != NULL) {
                    // based on identity assign to proper module holders
                    id = module->Identify();
                    if(id.compare("ISkyNetClassificationProtocol") == 0) {
                        this->m_classifiers.push_back({module,libHandle});
                        info = "    " + module->About() + " [OK]\n";
                        SKYNET_INFO(info.c_str() );
                    }
                }
                break;
            // directory , unless it is "." or ".." , copy names onto stack
            case DT_DIR:
                entryName = entry->d_name;
                if( (entryName.compare(".") != 0) && (entryName.compare("..") != 0) ) {
                    // entry's name is given relatively to already open's directory name
                    // so if this is another directory we need to concatenate both names
                    this->LoadModules(modulesDirectoryName + std::string("/") + entryName);
                }
                break;
            default:
                // ignore anything irrelevant
                break;
            }
        }
        closedir(modules_dir);
    }
}

