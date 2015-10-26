#include <dirent.h>
#include <getopt.h>
#include <errno.h>
#include <stack>
#include <iostream>
#include <fstream>

#include "skynet.h"
#include "os_inc.h"
#include "tests/randomPointsClassification.h"
#include "tests/tests.h"

SkyNet::SkyNet(int argc, char *const *argv) :  m_terminated(false), m_printmodules(false), m_printTests(false), m_enableModule(0), 
                                               m_moduleToLoad(""), m_testToExecute(0)  
{
    SKYNET_INFO("Skynet Initializing...\n\n");

    try {
        ProcessCommandLine(argc,argv);

        SKYNET_INFO("Initializing computing devices:\n");

        SKYNET_INFO("Loading Modules:\n");
        LoadModules(std::string("./modules") );
        LoadModules(std::string("/usr/share/skynet/modules") );
        PrintModules();
        PrintTests();
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
const std::vector<SkyNet::classificationModule>& SkyNet::getClassificationModules(void)    
{
    static std::vector<SkyNet::classificationModule> classifier;
    classifier.clear();

    // if no specific module was pointed out, then
    // return reference to list of all modules
    if(m_enableModule == 0) {
        return m_classifiers;
    } else {
        classifier.push_back(m_classifiers[m_enableModule - 1]);
        return classifier;
    }
}
//////////////////////////////////////////////////////////////////// 
const std::string& SkyNet::getMnistDataDir(void)
{
    return m_mnist_dir;
}
//////////////////////////////////////////////////////////////////// 
void SkyNet::PrintTests()
{
    if(m_printTests == true)
    {
        SKYNET_INFO("\n List of found Machine learning tests:\n\n");
        TestsRegistry::inst().printTests(); 
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
    printf("SkyNet [--help] [--list_tests] [--list_modules] [--resume=<Path to file with stored weights eg. final_weights.txt>]  [--module=<number of module to be loaded>]  [--test=<number of test to be executed>] [--mnist-data=<directory where MNIST datas are.>]\n");
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
                                 {"resume", required_argument, nullptr, 8 },
                                 {"test", required_argument, nullptr, 16 },
                                 {"list_tests", no_argument, nullptr, 32 },
                                 {"mnist-data", required_argument, nullptr, 64 },
                                 {0,0,0,0}};
    do
    {
        c = getopt_long(argc,argv, "", longopts, nullptr);
        // Unrecognized option?
        if(c == '?')
        {
            throw std::string("Skynet Error: Unrecognized option\n");
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
        } else if (c==8) {
            m_moduleToLoad = getModuleToLoad(optarg);
            SKYNET_INFO("Resuming Training for: %s\n",m_moduleToLoad.c_str());
        } else if (c==16) {
            m_testToExecute = std::stoi(optarg);
        } else if (c==32) {
            m_printTests = true;
            m_terminated = true;
        } else if (c==64) {
            m_mnist_dir = optarg;
        }
    }
    while(c != -1);

    return;
}
//////////////////////////////////////////////////////////////////// 
std::string SkyNet::getModuleToLoad(char *fileToLoad)
{
    std::ifstream file(fileToLoad);
    std::string module_name;
    if (file.is_open())
    {
        // Read name of module for which weights are stored
        std::getline(file,module_name);
        // read weights
        std::string weight;
        while(!file.eof()) {
            std::getline(file,weight,' ');
            // At very beginning there is a space so we may get empty read
            if(weight.size() != 0) {
                m_moduleWeights.push_back(std::stof(weight)); 
            }
        }
    } else {
        SKYNET_INFO(" %s",fileToLoad); 
        std::string err_string("Error: Unable to open a file: ");
        err_string += fileToLoad;
        throw std::invalid_argument(err_string);
    }
    // Return first line read starting from second character (to omit '#' that starts the line in a file)
    return module_name.substr(1); 
}
//////////////////////////////////////////////////////////////////// 
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
                module    = SkyNetOS::LoadModule(modulesDirectoryName + std::string("/") + entryName,&libHandle);
                if(module != NULL) {
                    // based on identity assign to proper module holders
                    id = module->Identify();
                    if(id.compare("ISkyNetClassificationProtocol") == 0) {
                        // If either no module to Load is specifically selected (All found modules are to be load)
                        // or module to Load was selected and candiadte module's about string matches
                        // then load candidate module
                        if((m_moduleToLoad.size() == 0) || ( m_moduleToLoad.compare(module->About()) == 0 ))  {
                            this->m_classifiers.push_back({module,libHandle});
                            // Initialize weights to those read from given file eg. resume ceased learning
                            if( m_moduleToLoad.compare(module->About()) == 0 ) {
                                module->setWeights(m_moduleWeights);
                            }
                            info = "    " + module->About() + " [OK]\n";
                            
                            SKYNET_INFO(info.c_str() );
                        }
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

    TestsRegistry::inst().executeTest(*this, m_testToExecute);
   
}

