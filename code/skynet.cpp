#include "skynet.h"
#include <dirent.h>
#include <errno.h>


SkyNet::SkyNet(void){
}

SkyNet::~SkyNet()
{
}

/* Scan directory with modules and loadem all */
void SkyNet::LoadModules(std::string modulesDirectoryName)
{
    // search for files with extension snm
    DIR   *modules_dir;
    dirent* entry;

    //TODO: Parse subdirectories to get list of *.so files (modules)

    modules_dir = opendir(modulesDirectoryName.c_str() );
    if(modules_dir != NULL) {

        while( (entry = readdir(modules_dir) ) != NULL) {
            printf("Potential module: %s\n",entry->d_name);
        }
        closedir(modules_dir);
    } else {
        perror("Modules initialization unsuccessful");
    }
}

