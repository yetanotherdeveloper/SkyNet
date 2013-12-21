#include <dirent.h>
#include <errno.h>
#include <stack>

#include "skynet.h"
#include "os_inc.h"


SkyNet::SkyNet(void)
{
    SKYNET_INFO("Skynet Initializing...\n\n");

    SKYNET_INFO("Loading Modules:\n");
    LoadModules(std::string("./modules"));
}

SkyNet::~SkyNet()
{
    std::vector<classificationModule>::iterator it;
    //TODO: release all modules and its libs handles
    for(it = m_classifiers.begin(); it != m_classifiers.end(); ++it) 
    {
        SkyNetOS::ReleaseModule(&(it->module),&(it->libHandle));
    }
}

/* Scan directory with modules and loadem all */
void SkyNet::LoadModules(std::string modulesDirectoryName)
{
    // search for files with extension snm
    DIR   *modules_dir;
    dirent* entry;
    std::string entryName;
    ISkyNetClassificationProtocol* module;
    void* libHandle;
    std::string id;
    std::string info;


    modules_dir = opendir(modulesDirectoryName.c_str() );
    if(modules_dir != NULL) {
        while( (entry = readdir(modules_dir) ) != NULL) {
            SKYNET_DEBUG("Potential module: %s\n",entry->d_name);
            switch(entry->d_type){
                // regular file, check if this is DSO and load
                case DT_REG:
                    entryName = entry->d_name;
                    module = SkyNetOS::LoadModule(modulesDirectoryName + std::string("/") + entryName,&libHandle);
                    if(module != NULL) {
                        // based on identity assign to proper module holders
                        id = module->Identify();
                        if(id.compare("ISkyNetClassificationProtocol") == 0) {
                            this->m_classifiers.push_back({module,libHandle}); 
                            info = "    " + module->About() + " [OK]\n";        
                            SKYNET_INFO(info.c_str());
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

