#ifndef __TESTS__
#define __TESTS__

#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

// Callback function , that each test should be providing  
typedef void (*runTestCallback)(void);

// Here I'm writting singleton object for keeping tests
class TestsRegistry
{
public:
    static TestsRegistry& inst(void)
    {
        static TestsRegistry instance;                
        return instance;
    }

    bool addTest(std::string testName, runTestCallback start_test_routine)
    {
        m_testsRegistry.push_back(std::make_pair(testName,start_test_routine)); 
        return true;
    }

    void printTests()
    {
        unsigned short i = 1;
        for(auto& registered_test : m_testsRegistry) 
        {
            std::cout << i++ << ". "<< registered_test.first << std::endl;
        }
    }

    void executeTest(unsigned short test_id = 0)
    {
        if(test_id == 0)
        {
            // Execute all tests
            for(auto& test_to_execute : m_testsRegistry)
            {
                test_to_execute.second();
            }
        } else {
            // Execute selected test
            // test_id is from 1 to number of tests
            // so we need to subtract one to get actual entry in registry
            assert(test_id <= m_testsRegistry.size());
            m_testsRegistry[test_id-1].second();
        }    
    }

private:
    TestsRegistry()
    {
        m_testsRegistry.clear();
    }

    std::vector<std::pair<std::string,runTestCallback>> m_testsRegistry;
};

#endif
