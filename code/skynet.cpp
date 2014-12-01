#include <dirent.h>
#include <getopt.h>
#include <errno.h>
#include <stack>
#include <CL/cl.hpp>

#include "skynet.h"
#include "os_inc.h"
#include "tests/randomPointsClassification.h"

SkyNet::SkyNet(int argc, char *const *argv) :  m_terminated(false), m_printmodules(false), m_enableModule(0) 
{
    SKYNET_INFO("Skynet Initializing...\n\n");

    SKYNET_INFO("Processing command line\n");
    try {
        ProcessCommandLine(argc,argv);

        SKYNET_INFO("Initializing computing devices:\n");
        InitDevices();

        SKYNET_INFO("Loading Modules:\n");
        LoadModules(std::string("./modules") );
        LoadModules(std::string("/usr/share/skynet/modules") );
        PrintModules();
    }
    catch(std::string err)
    {
        SKYNET_INFO(err.c_str() );
        m_terminated = true;
    }
    // CLI argument conversion to numeric value , failed
    catch(std::invalid_argument err)
    {
        SKYNET_INFO("SkyNet Error: Invalid Argument value: %s\n", err.what());
        m_terminated = true;
    }
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
//////////////////////////////////////////////////////////////////// 
void SkyNet::PrintModules()
{
    if(m_printmodules == true)
    {
        SKYNET_INFO("\n List of found Machine learning modules:\n\n");
        int lpr = 1;
        for(auto it = m_classifiers.begin(); it < m_classifiers.end(); ++it) {
            SKYNET_INFO("    %d. %s\n",lpr++,it->module->About().c_str() );
        }
    }
}
//////////////////////////////////////////////////////////////////// 
void SkyNet::PrintHelp()
{
    printf("SkyNet [--help] [--list_modules]\n");
    return;
}
//////////////////////////////////////////////////////////////////// 
// Process commandline
void SkyNet::ProcessCommandLine(int argc, char *const *argv)
{
    int c;

    // Wez tutaj okresla tablice opcji:
    // 1. Help (--help)
    // 2. Test (--test , bez argumentow wyswietla wszystkie testy z nazwy, poza tym all i z listy dostepnych modulow)
    // 3. Dodaj Unit testy na linie polecen

    struct option longopts[] = { {"help", no_argument, nullptr, 1 },
                                 {"module", required_argument, nullptr, 2 },
                                 {"list_modules", no_argument, nullptr, 4 },
                                 {0,0,0,0}};
    do
    {
        c = getopt_long(argc,argv, "", longopts, nullptr);
        // Unrecognized option?
        if(c == '?')
        {
            throw std::string("Skynet Error: Unrecognized option");
        }
        else if(c == 1)
        {
            PrintHelp();
            throw std::string("");
        } else if(c == 2) 
        {
            m_enableModule = std::stoi(optarg);         
        } else if (c==4) {
            m_printmodules = true;
            m_terminated = true;
        }
    }
    while(c != -1);

    return;
}
//////////////////////////////////////////////////////////////////// 


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
        SKYNET_DEBUG("ERROR: No computing platforms found!\n");
        return;
    }

    // List platforms capabilities

    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it ) {

        plat_it->getInfo(CL_PLATFORM_NAME, &platform_param_value);
        SKYNET_DEBUG("\nPlatform: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_VENDOR, &platform_param_value);
        SKYNET_DEBUG("   vendor: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_VERSION, &platform_param_value);
        SKYNET_DEBUG("   version: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_PROFILE, &platform_param_value);
        SKYNET_DEBUG("   profile: %s\n",platform_param_value.c_str());

        plat_it->getInfo(CL_PLATFORM_EXTENSIONS, &platform_param_value);
        SKYNET_DEBUG("   extensions: %s\n",platform_param_value.c_str());

        // Now for given platform get list of devices and their capabilities

        if( plat_it->getDevices(CL_DEVICE_TYPE_ALL, &devices ) == CL_SUCCESS ) {
        
            for(std::vector<cl::Device>::iterator device_it = devices.begin(); device_it != devices.end(); ++device_it) {
                device_it->getInfo(CL_DEVICE_NAME, &device_param_string_value);
                SKYNET_DEBUG("   Device: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_VENDOR, &device_param_string_value);
                SKYNET_DEBUG("       vendor: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_EXTENSIONS, &device_param_string_value);
                SKYNET_DEBUG("       extensions: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_VERSION, &device_param_string_value);
                SKYNET_DEBUG("       device version: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DRIVER_VERSION, &device_param_string_value);
                SKYNET_DEBUG("       driver version: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_OPENCL_C_VERSION, &device_param_string_value);
                SKYNET_DEBUG("       OpenCL C Version: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_PROFILE, &device_param_string_value);
                SKYNET_DEBUG("       profile: %s\n",device_param_string_value.c_str());

                device_it->getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
                SKYNET_DEBUG("       global memory size: %lu\n",global_mem_size);

                device_it->getInfo(CL_DEVICE_ADDRESS_BITS, &addr_size);
                SKYNET_DEBUG("       address space size: %d\n",addr_size);

                device_it->getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &device_param_numeric_value);
                SKYNET_DEBUG("       max_work_item_sizes: %d x %d x %d\n",device_param_numeric_value[0],device_param_numeric_value[1],device_param_numeric_value[2]);

                device_it->getInfo(CL_DEVICE_AVAILABLE, &device_param_bool_value);
                SKYNET_DEBUG("       availability: %d\n",device_param_bool_value);

                device_it->getInfo(CL_DEVICE_COMPILER_AVAILABLE, &device_param_bool_value);
                SKYNET_DEBUG("       compiler availability: %d\n",device_param_bool_value);

                device_it->getInfo(CL_DEVICE_ADDRESS_BITS, &addr_size);
                SKYNET_DEBUG("       address space size: %d\n",addr_size);
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
            (m_devices.begin())->getInfo(CL_DEVICE_NAME, &device_param_string_value);
            SKYNET_DEBUG(" HARDCODED GPU DEVICE to be used (temporary)  Device: %s\n",device_param_string_value.c_str());
            return;
        }
    }
    // If no GPU device is found then pick anything
    for(std::vector<cl::Platform>::iterator plat_it = platforms.begin(); plat_it != platforms.end(); ++plat_it ) {
        if( plat_it->getDevices(CL_DEVICE_TYPE_ALL, &m_devices ) == CL_SUCCESS ) {
            (m_devices.begin())->getInfo(CL_DEVICE_NAME, &device_param_string_value);
            SKYNET_DEBUG(" Chosen DEVICE to be used (temporary)  Device: %s\n",device_param_string_value.c_str());
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
            //SKYNET_DEBUG("Potential module: %s\n",entry->d_name);
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
    // if termination flag is on then do not even start work
    if (m_terminated == true)
    {
        return;
    }
    // diagnostic results are stored in directory named after process ID
    SkyNetDiagnostic diagnostic;
    // Just run all tests
    randomPointsClassification rpc(100,2);
    std::vector<classificationModule>::iterator it;
    unsigned short                              module_lpr = 1;
    for(it = m_classifiers.begin(); it != m_classifiers.end(); ++it) {
        // Run all modules or only then one indicated by cli option: --module
        if( (m_enableModule == 0) || (module_lpr == m_enableModule) )
        {
            diagnostic.reset();
            SKYNET_INFO("Running OCL test against: %s\n",it->module->About().c_str() );
            // Pass Input data , and initial weights to RunCL , RunRef functions
            //it->module->RunCL();
            rpc.setWeights(it->module->RunRef(rpc.getTrainingData(),diagnostic) );
            SKYNET_INFO("In-sample error: %f Out-of-sample error: %f\n",  rpc.validate(it->module->getClassification(rpc.getTrainingData() ) ),rpc.verify(it->module->getClassification(rpc.getTestingData() ) ) );
            SKYNET_INFO("GetError: %f\n",it->module->getError(rpc.getTrainingData() ) );
            diagnostic.makeWeightsAnalysis(it->module->About() );
            diagnostic.makeTrainingAnalysis(it->module->About(),rpc.getTrainingData(), rpc.getTargetWeights(),rpc.getWeights() );
            diagnostic.makeGeneralizationAnalysis(it->module->About(),rpc.getTestingData(), rpc.getTargetWeights(),rpc.getWeights() );
        }
        ++module_lpr;
    }
}

