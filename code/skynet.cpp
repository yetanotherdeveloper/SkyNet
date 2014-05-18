#include <dirent.h>
#include <errno.h>
#include <stack>
#include <CL/cl.hpp>

#include "skynet.h"
#include "os_inc.h"
#include "tests/randomPointsClassification.h"

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
    std::string device_param_string_value;    
    std::vector<size_t> device_param_numeric_value;    
    cl_device_id device;
    cl_ulong global_mem_size;
    cl_uint addr_size;
    cl_bool device_param_bool_value;

    if( cl::Platform::get(&platforms) != CL_SUCCESS) {
        SKYNET_INFO("ERROR: No computing platforms found!\n");
        return;
    }

    // List platforms capabilities

    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it ) {

        plat_it->getInfo(CL_PLATFORM_NAME, &platform_param_value);
        SKYNET_INFO("\nPlatform: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_VENDOR, &platform_param_value);
        SKYNET_INFO("   vendor: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_VERSION, &platform_param_value);
        SKYNET_INFO("   version: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_PROFILE, &platform_param_value);
        SKYNET_INFO("   profile: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_EXTENSIONS, &platform_param_value);
        SKYNET_INFO("   extensions: %s\n",platform_param_value.c_str());

        // Now for given platform get list of devices and their capabilities

        if( plat_it->getDevices(CL_DEVICE_TYPE_ALL, &devices ) == CL_SUCCESS ) {
        
            for(std::vector<cl::Device>::iterator device_it = devices.begin(); device_it != devices.end(); ++device_it) {
                device_it->getInfo(CL_DEVICE_NAME, &device_param_string_value);
                SKYNET_INFO("   Device: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_VENDOR, &device_param_string_value);
                SKYNET_INFO("       vendor: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_EXTENSIONS, &device_param_string_value);
                SKYNET_INFO("       extensions: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_VERSION, &device_param_string_value);
                SKYNET_INFO("       device version: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DRIVER_VERSION, &device_param_string_value);
                SKYNET_INFO("       driver version: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_OPENCL_C_VERSION, &device_param_string_value);
                SKYNET_INFO("       OpenCL C Version: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_PROFILE, &device_param_string_value);
                SKYNET_INFO("       profile: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
                SKYNET_INFO("       global memory size: %lu\n",global_mem_size);

                device_it->getInfo(CL_DEVICE_ADDRESS_BITS, &addr_size);
                SKYNET_INFO("       address space size: %d\n",addr_size);

                device_it->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &device_param_numeric_value);
                SKYNET_INFO("       max_work_item_sizes: %d x %d x %d\n",device_param_numeric_value[0],device_param_numeric_value[1],device_param_numeric_value[2]);

                device_it->getInfo(CL_DEVICE_AVAILABLE, &device_param_bool_value);
                SKYNET_INFO("       availability: %d\n",device_param_bool_value);

                device_it->getInfo(CL_DEVICE_COMPILER_AVAILABLE, &device_param_bool_value);
                SKYNET_INFO("       compiler availability: %d\n",device_param_bool_value);

                device_it->getInfo(CL_DEVICE_ADDRESS_BITS, &addr_size);
                SKYNET_INFO("       address space size: %d\n",addr_size);
            }
   

 
        } else {
            SKYNET_DEBUG("      No Devices found at platform!\n");
        }

    }

    // temporary get Any GPU device to run some tests
    // lateron will change it to running tasks against all devices (separate threads), for comparison
    // or selected device basedo n commandline
    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it ) {

        if( plat_it->getDevices(CL_DEVICE_TYPE_GPU, &m_devices ) == CL_SUCCESS ) {
            (devices.begin())->getInfo(CL_DEVICE_NAME, &device_param_string_value);
            SKYNET_INFO(" HARDCODED GPU DEVICE to be usoed (temporary)  Device: %s\n",device_param_string_value.c_str());
            return;
        }
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
                module    = SkyNetOS::LoadModule(modulesDirectoryName + std::string("/") + entryName,&libHandle,&(m_devices[0]));
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

void SkyNet::RunTests()
{
    // Just run all tests
    randomPointsClassification rpc(100);
    std::vector<classificationModule>::iterator it; 
    for(it = m_classifiers.begin(); it != m_classifiers.end(); ++it) {
        printf("Running OCL test against: %s\n",it->module->About().c_str());
        // Pass Input data , and initial weights to RunCL , RunRef functions 
        //it->module->RunCL();
        it->module->RunRef(rpc.getTrainingData());
    }
}

