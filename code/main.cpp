#include <string>
#include "protocol.h"
#include "skynet.h"
#include "os_inc.h"


int main()
{
    // Parse CommandLine
    SkyNet m_skynet;
  
    // Run tests, all or selected ones
    m_skynet.RunTests(); 
    
    return 0;
}
