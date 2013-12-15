#include "protocol.h"
//#include "os_inc.h"
#include "skynet.h"
#include <string>



int main()
{
    SkyNet m_skynet;
    // Parse CommandLine
   
    // Load Libraries 
    m_skynet.LoadModules(std::string("./modules"));
    
    return 0;
}
