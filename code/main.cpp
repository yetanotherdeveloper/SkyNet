#include <string>
#include "protocol.h"
#include "skynet.h"
#include "os_inc.h"


int main(int argc, char **argv)
{
    // Parse CommandLine
    SkyNet m_skynet(argc, argv);
  
    // Run tests, all or selected ones
    m_skynet.RunTests(); 
    
    return 0;
}
