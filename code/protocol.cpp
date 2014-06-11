#include "protocol.h"
#include "os_inc.h"
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>

cl_int SkyNetOpenCLHelper::err = 0;

// Function to create context, handle OpenCL errors
std::unique_ptr<cl::Context> SkyNetOpenCLHelper::createCLContext(const cl::Device* pdevice)
{
    // make a context , command queue
    std::unique_ptr<cl::Context> pcontext(new cl::Context(*pdevice,NULL,NULL,NULL,&err) );

    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL context: %d\n",err);
        printf("Error creating OpenCL context : %d\n",err);
    }
    return pcontext;
}

std::unique_ptr<cl::CommandQueue> SkyNetOpenCLHelper::createCLCommandQueue(const cl::Context& context, const cl::Device& device)
{
    std::unique_ptr<cl::CommandQueue> queue(new cl::CommandQueue(context,device,0,&err) );

    if( err != CL_SUCCESS )
    {
        printf(" Error creating OpenCL command queue: %d\n",err);
    }
    return queue;
}

std::unique_ptr<cl::Kernel> SkyNetOpenCLHelper::makeKernels(const cl::Context &context, const cl::Device &device, const std::string & kernelSource, const std::string kernelName )
{
    cl::Program::Sources kern_sources(1, std::make_pair(kernelSource.c_str(), kernelSource.length() + 1) );
    cl::Program program(context, kern_sources,&err);
    if( err != CL_SUCCESS )
    {
        printf(" Error creating OpenCL program from source: %d\n",err);
        return std::unique_ptr<cl::Kernel>(nullptr);
    }

    std::vector<cl::Device> targetDevices(1,device);
    std::string dev_version = device.getInfo<CL_DEVICE_VERSION>();

    std::string buildOptions("-cl-std=CL");
    // Parse Device Version string to see what CL is supported.
    // eg. format of string is :"OpenCL major_number.minor_number"
    // get 3 characters eg. "major.minor" and append them to options
    buildOptions.append(dev_version.substr(7,3) );

    err = program.build(targetDevices,buildOptions.c_str() );
    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error Building OpenCL program failed: %d\n",err);
        std::string log;
        err = program.getBuildInfo(device,CL_PROGRAM_BUILD_LOG,&log);
        printf(" OpenCL program build log: %s\n",log.c_str() );
        return std::unique_ptr<cl::Kernel>(nullptr);
    }

    std::unique_ptr<cl::Kernel> kernel(new cl::Kernel(program, kernelName.c_str(), &err) );
    if( err != CL_SUCCESS )
    {
        //SKYNET_INFO(" Error creating OpenCL command queue: %d\n",err);
        printf(" Error creating OpenCL PLA kernel : %d\n",err);
        return std::unique_ptr<cl::Kernel>(nullptr);
    }

    return kernel;
}

//---> SkyNetDiagnostic definitions
SkyNetDiagnostic::SkyNetDiagnostic()
{
    m_dumpDirName = std::to_string(SkyNetOS::getPID() );
}


SkyNetDiagnostic::~SkyNetDiagnostic()
{
}


void SkyNetDiagnostic::storeWeights(const std::vector<float> &weights)
{
    m_historyOfWeights.push_back(weights);
}


// TODO: portable way of creating directories
void SkyNetDiagnostic::dumpWeights(const std::string& dirName)
{
    if(m_historyOfWeights.empty() )
    {
        printf("Error: There is no weights to be dumped\n");
        return;
    }
    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName);

    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName + "/" + dirName);

    // dump weights with proper comments ofcourse as a first line
    std::ofstream dumpfile(m_dumpDirName + "/" + dirName + "/weights.txt", std::ios::trunc);
    dumpfile << "#";
    // Knowing that weights history contains some entries
    // we check the first entry to see how many weights was there
    unsigned int nr_weights = m_historyOfWeights[0].size();
    for(unsigned int i = 0; i < nr_weights; ++i)
    {
        dumpfile << " w" + std::to_string(i);
    }
    dumpfile << std::endl;

    // This module holds the history of weights (how weights evolved over time
    // number of weights is constant across history
    for(unsigned int i = 0; i < m_historyOfWeights.size(); ++i)
    {
        for(unsigned int j = 0; j < nr_weights; ++j)
        {
            dumpfile << " " + std::to_string( (m_historyOfWeights[i])[j]);
        }
        dumpfile << std::endl;
    }
}


void SkyNetDiagnostic::makeAnalysis(const std::string& dirName,const std::string& dataFilename, const std::string& scriptFilename,
                                            const std::vector<point> & set, const std::vector<float> &targetWeights,
                                            const std::vector<float> &learnedWeights)
{
    if(learnedWeights.empty()) {
        SKYNET_DEBUG("Warning: No learned weights send to makeTrainingAnalysis function\n");         
        return;
    }

    // Dump training data into file
    
    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName);

    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName + "/" + dirName);

    // dump weights with proper comments ofcourse as a first line
    std::ofstream dumpfile(m_dumpDirName + "/" + dirName + "/" + dataFilename, std::ios::trunc);
    dumpfile << "#x y class" << std::endl;

    for(unsigned int i=0; i < set.size(); ++i)
    {
        dumpfile << set[i].x << " " << set[i].y << " " << set[i].classification << std::endl;
    }
    // Generate gnuplot script drawing a validation chart
    // presenting points as well as target function and learned(trained) function

    std::string desc = scriptFilename.substr(0,scriptFilename.rfind("."));    

    // Make something like: plot "trainingData.txt" using 1:($3 == 1 ?  $2:1/0), "trainingData.txt" using 1:($3 == -1? $2:1/0)
    std::ofstream script(m_dumpDirName + "/" + dirName + "/" + scriptFilename, std::ios::trunc);
    script << "set terminal png size 1280,960"<< std::endl;
    script << "set output \""<< desc.c_str()<<".png\""<< std::endl;
    script << "set title \"In-sample error (" << desc.c_str() << ")\"" << std::endl;
    script << "plot \"" << dataFilename.c_str() << "\" using 1:($3 == 1 ?  $2:1/0) title \"" << desc.c_str()<< " set (class +1)\" , \
             \"" << dataFilename.c_str() << "\" using 1:($3 == -1? $2:1/0) title \"" << desc.c_str()<<" set (class -1)\"";
    
    // Make transformation (assuming w2 <> 0) w0 + w1*x w2*y = 0 <=> y = -w1/w2 *x -w0/w2
    script << "," << -(targetWeights[1]/targetWeights[2]) << "*x +" << -(targetWeights[0]/targetWeights[2]) << " title \"target function\"";
    script << "," << -(learnedWeights[1]/learnedWeights[2]) << "*x +" << -(learnedWeights[0]/learnedWeights[2]) << " title \"learned function\""<< std::endl;
    
}


void SkyNetDiagnostic::makeTrainingAnalysis(const std::string& dirName,const std::vector<point> & set,
                                            const std::vector<float> &targetWeights,const std::vector<float> &learnedWeights)
{
        makeAnalysis(dirName,"trainingData.txt","validation.plot",set,targetWeights,learnedWeights);
}


void SkyNetDiagnostic::makeGeneralizationAnalysis(const std::string& dirName,const std::vector<point> & set,
                                            const std::vector<float> &targetWeights,const std::vector<float> &learnedWeights)
{
        makeAnalysis(dirName,"testingData.txt","verification.plot",set,targetWeights,learnedWeights);
}