#ifndef __TESTS__
#define __TESTS__

#include <iostream>
#include <vector>

// Here I'm writting singleton object for keeping tests
class TestsRegistry
{
public:
    static TestsRegistry& inst(void)
    {
        static TestsRegistry instance;                
        return instance;
    }

    bool addTest(std::string testName)
    {
        m_testsRegistry.push_back(testName); 
        return true;
    }

    void printTests()
    {
        for(auto& test: m_testsRegistry ) 
        {
            std::cout << test << std::endl;
        }
    }

private:
    TestsRegistry()
    {
        m_testsRegistry.clear();
    }

    std::vector<std::string> m_testsRegistry;
};

#endif
