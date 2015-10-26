#include "protocol.h"
#include "os_inc.h"
#include <iostream>
#include <fstream>

//---> SkyNetDiagnostic definitions
SkyNetDiagnostic::SkyNetDiagnostic()
{
    m_dumpDirName =  SkyNetOS::GetHomeDirectory() + "/skynet_logs/";
    SkyNetOS::CreateDirectory(m_dumpDirName);
    m_dumpDirName += std::to_string(SkyNetOS::getPID() );
    SkyNetOS::CreateDirectory(m_dumpDirName);
}


SkyNetDiagnostic::~SkyNetDiagnostic()
{
}


void SkyNetDiagnostic::reset()
{
    m_history.clear();
}


void SkyNetDiagnostic::storeWeightsAndError(const std::vector<float> &weights, float in_error, float val_error)
{
    m_history.push_back(historicalNote(weights,in_error, val_error) );
}
///////////////////////////////////////////////////////////////////////////////////////
/// Function to save learned wieghts into file: final_weights.txt
void SkyNetDiagnostic::saveWeightsToFile(const std::string& dirName)
{
    if(m_history.empty() )
    {
        printf("Error: There is no weights to be dumped\n");
        return;
    }

    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName + "/" + dirName);

    // dump weights with proper comments ofcourse as a first line
    std::ofstream dumpfile(m_dumpDirName + "/" + dirName + "/final_weights.txt", std::ios::trunc);
    dumpfile << "#" << dirName << std::endl;

    unsigned int nr_weights = m_history[m_history.size() - 1].m_weights.size(); 

    for(unsigned int j = 0; j < nr_weights; ++j)
    {
        dumpfile << " " + std::to_string( (m_history[m_history.size() - 1].m_weights)[j]);
    }
}
///////////////////////////////////////////////////////////////////////////////////////
// TODO: portable way of creating directories
void SkyNetDiagnostic::makeWeightsAnalysis(const std::string& dirName)
{
    if(m_history.empty() )
    {
        printf("Error: There is no weights to be dumped\n");
        return;
    }

    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName + "/" + dirName);

    // dump weights with proper comments ofcourse as a first line
    std::ofstream dumpfile(m_dumpDirName + "/" + dirName + "/weights_and_errors.txt", std::ios::trunc);
    dumpfile << "#";
    // Knowing that weights history contains some entries
    // we check the first entry to see how many weights was there
    unsigned int nr_weights = m_history[0].m_weights.size();
    for(unsigned int i = 0; i < nr_weights; ++i)
    {
        dumpfile << " w" + std::to_string(i);
    }
    // After weights there is an in sample error
    // and validation error as a next one
    dumpfile << " Ein" << " Eval" << std::endl;

    // This module holds the history of weights (how weights evolved over time
    // number of weights is constant across history
    for(unsigned int i = 0; i < m_history.size(); ++i)
    {
        for(unsigned int j = 0; j < nr_weights; ++j)
        {
            dumpfile << " " + std::to_string( (m_history[i].m_weights)[j]);
        }
        dumpfile << " " << std::to_string( m_history[i].m_in_error) << " " << std::to_string(m_history[i].m_val_error) <<  std::endl;
    }

    std::ofstream script(m_dumpDirName + "/" + dirName + "/" + "/weights_and_errors.plot", std::ios::trunc);
    script << "set terminal png size 1280,960" << std::endl;
    script << "set xlabel \"Iteration\"" << std::endl;
    script << "set ylabel \"Error\"" << std::endl;
    script << "set output \"" << "weights_and_errors.png\"" << std::endl;
    script << "set title \"In-sample and Validation errors: E_in(iteration), E_val(iteration)"  << std::endl;
    script << "plot \"" << "weights_and_errors.txt" << "\" using "<<  std::to_string(m_history[0].m_weights.size() + 1) << " title \"In sample Error\"," << "\"" << "weights_and_errors.txt" << "\" using "<<  std::to_string(m_history[0].m_weights.size() + 2) << " title \"Validation Error\""<< std::endl;
    script << "set terminal x11" << std::endl;
    script << "set output " << std::endl;
    script << "replot" << std::endl;
}


void SkyNetDiagnostic::makeAnalysis(const std::string& dirName,const std::string& dataFilename, const std::string& scriptFilename,
                                    const std::vector<point> & set, const std::vector<float> &targetWeights,
                                    const std::vector<float> &learnedWeights)
{
    if(learnedWeights.empty() )
    {
        SKYNET_DEBUG("Warning: No learned weights send to makeTrainingAnalysis function\n");
        return;
    }

    // Dump training data into file
    // create PID directory, do some check if this is allowed
    SkyNetOS::CreateDirectory(m_dumpDirName + "/" + dirName);

    // dump weights with proper comments ofcourse as a first line
    std::ofstream dumpfile(m_dumpDirName + "/" + dirName + "/" + dataFilename, std::ios::trunc);
    dumpfile << "#x y class" << std::endl;

    for(unsigned int i = 0; i < set.size(); ++i)
    {
        dumpfile << set[i].x << " " << set[i].y << " " << set[i].classification << std::endl;
    }
    // Generate gnuplot script drawing a validation chart
    // presenting points as well as target function and learned(trained) function

    std::string desc = scriptFilename.substr(0,scriptFilename.rfind(".") );

    // Make something like: plot "trainingData.txt" using 1:($3 == 1 ?  $2:1/0), "trainingData.txt" using 1:($3 == -1? $2:1/0)
    std::ofstream script(m_dumpDirName + "/" + dirName + "/" + scriptFilename, std::ios::trunc);
    script << "set terminal png size 1280,960" << std::endl;
    script << "set output \"" << desc.c_str() << ".png\"" << std::endl;
    script << "set title \"In-sample error (" << desc.c_str() << ")\"" << std::endl;
    script << "plot \"" << dataFilename.c_str() << "\" using 1:($3 == 1 ?  $2:1/0) title \"" << desc.c_str() << " set (class +1)\" , \
             \"" << dataFilename.c_str() << "\" using 1:($3 == -1? $2:1/0) title \"" << desc.c_str() << " set (class -1)\"";

    //TODO: Somehow print info about not recognized points

    // Make transformation (assuming w2 <> 0) w0 + w1*x w2*y = 0 <=> y = -w1/w2 *x -w0/w2
    //script << "," << -(targetWeights[1]/targetWeights[2]) << "*x +" << -(targetWeights[0]/targetWeights[2]) << " title \"target function\"";
    //script << "," << -(learnedWeights[1]/learnedWeights[2]) << "*x +" << -(learnedWeights[0]/learnedWeights[2]) << " title \"learned function\""<< std::endl;

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
