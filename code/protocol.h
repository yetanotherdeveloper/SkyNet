#ifndef __SKYNET_PROTOCOL__
#define __SKYNET_PROTOCOL__
#include <iostream>
#include <termios.h>    // Termios 
#include <fcntl.h>      // file control
#include <string>
#include <memory>
#include <vector>

// Note: Module should export following function:
//extern "C" ISkyNetClassificationProtocol* CreateModule(const cl::Device* const pdevice)



class SkyNetDiagnostic
{
    struct historicalNote
    {
        std::vector<float> m_weights;
        float m_in_error;
        float m_val_error;
        historicalNote(const std::vector<float> &weights, float in_error, float val_error) :
                             m_weights(weights), m_in_error(in_error), m_val_error(val_error) 
        {
        }

    };
private:
    std::vector<historicalNote> m_history;                /// Weights assuming that in chronological
                                                          /// order First one is the oldest(earliest) iteration
    std::string m_dumpDirName;                             /// Directory where gathered/generated data will be stored
public:
    SkyNetDiagnostic();
    ~SkyNetDiagnostic();
    void reset();
    void storeWeightsAndError(const std::vector<float> &weights, float in_error, float val_error);
    void makeWeightsAnalysis(const std::string& dirName);
    void saveWeightsToFile(const std::string& dirName);
    void makeTrainingAnalysis(const std::string& dirName,
                              const std::vector<std::vector<float>> & setData,
                              const std::vector<int>  &setLabels,
                              const std::vector<float> &targetWeights,
                              const std::vector<float> &learnedWeights);
    void makeGeneralizationAnalysis(const std::string& dirName,
                                    const std::vector<std::vector<float>> & setData,
                                    const std::vector<int>  &setLabels,
                                    const std::vector<float> &targetWeights,const std::vector<float> &learnedWeights);
private:
    void makeAnalysis(const std::string& dirName,const std::string& dataFilename, const std::string& scriptFilename,
                      const std::vector<std::vector<float>> & setData,
                      const std::vector<int>  &setLabels,
                      const std::vector<float> &targetWeights,
                      const std::vector<float> &learnedWeights);
};

typedef void (SkyNetDiagnostic::*fp_storeWeights)(const std::vector<float> &weights);

/// This module is implementing early stop algorithm
class SkyNetEarlyStop
{
private:
        std::vector<float> m_weights;
        float m_opt_val_error;
        unsigned int m_min_num_iterations; // minimum number of iterations to be processed (we won't stop before getting
                                           //  errors of that much iterations at least
        unsigned int m_max_num_iterations; //< maximum number of iterations to be executed (no more than this value is to be done)
        float m_alpha;  //threashold above which we stop training
        float m_beta;   //threshold below which training progress is considered not very high
        unsigned int m_strip_size; //< size of strip within which we evaluate training error progress
        std::vector<float> m_training_errors; //Holds all training elements of given strip
public:
    SkyNetEarlyStop(unsigned int min_num_iterations,
                    unsigned int max_num_iterations,
                    float alpha) :
                                    m_opt_val_error(1.0f),
                                    m_alpha(alpha), 
                                    m_min_num_iterations(min_num_iterations),
                                    m_max_num_iterations(max_num_iterations),
                                    m_strip_size(5),
                                    m_beta(0.1)  
    {
        m_training_errors.resize(m_strip_size, 1.0f);
    } 

    bool earlyStop( const unsigned int it, std::vector<float>& weights, float val_error, float training_error)
    {
        // If recent validation error is lower than optimal(minimal) one recorded
        // then make it new optimal validation error and store corressponding weights
        if(val_error < m_opt_val_error ) {
            m_weights = weights;
            m_opt_val_error = val_error;
        }

        // Store training errors from a current strip in a vector
        // to lateron calculate average / minimal ratio
        m_training_errors[it % m_strip_size] = training_error;

        // No more iterations than given maximum is to be executed
        if(it > m_max_num_iterations) {
          return true;
        }

        // 1. At least minimal number of iterations has to pass to even consider early stopping of training
        // 2. Training error drop shuld be at least equal to beta
        // 3. Make validation / training error validation at the end of strip
        // 4. Generalization loss has to be higher than threshold (alpha)
        if((it > m_min_num_iterations) && (it % m_strip_size == 0  ) ) {

            float in_err_drop = getTrainingRate(); 
            if( in_err_drop < m_beta )
            {
                printf("InErr drop: %f\n",in_err_drop);
                float gen_loss = val_error/m_opt_val_error - 1.0f;
                printf("GenLoss[%d]: %f\n",it,gen_loss);
                if(gen_loss > m_alpha) {
                    return true;
                }
            }
        } 

        return false;
    }

    void getOptimalWeights(std::vector< float > &optimal_weights)
    {
        optimal_weights = m_weights;
    }
private:
    // Get rate of average training error within a strip to minimum training
    // error within a strip (minus 1.0)
    // so it is measure of progress in training error evaluation
    float getTrainingRate(void)
    {
        float sum_errors = m_training_errors[0];
        float min_error = sum_errors;
        for( short int i = 1; i < m_strip_size; ++i)
        {
            sum_errors += m_training_errors[i];
            min_error = m_training_errors[i] < min_error ? m_training_errors[i] : min_error;
        } 
        return (sum_errors / (m_strip_size * min_error)) - 1.0f;
    }
};

/// This module can check if specific key was pressed 
/// if specific key was pressed then call operator returns true
class SkynetTerminalInterface
{
private:
    char           m_trigger_key;
    struct termios m_original, m_modified;
    int            m_stdio_flags;

public:
    SkynetTerminalInterface(char trigger_key) : m_trigger_key(trigger_key)
    {
        std::cout << "To cease learning and store current result press: " << m_trigger_key << " key." << std::endl;
    }

    bool operator()()
    {
        initTermios();
        char a = getchar();
        resetTermios();
        return a == m_trigger_key ? true : false;
    }

private:

    void initTermios(void)
    {
        tcgetattr(0,&m_original);
        m_modified          = m_original;
        m_modified.c_lflag &= ~ICANON;
        m_modified.c_lflag &= ~ECHO;
        tcsetattr(0, TCSANOW,&m_modified);
        m_stdio_flags = fcntl(0,F_GETFL,0);
        fcntl(0,F_SETFL, m_stdio_flags | O_NONBLOCK);
    }

    void resetTermios(void)
    {
        tcsetattr(0, TCSANOW, &m_original);
        fcntl(0,F_SETFL, m_stdio_flags );
    }
};

class ISkyNetClassificationProtocol
{
public:
    virtual ~ISkyNetClassificationProtocol() {
    };
    virtual const std::vector<float> & RunCL(const std::vector<std::vector<float>> &, SkyNetDiagnostic &, SkynetTerminalInterface& ) = 0;
    virtual void RunRef( const std::vector< std::vector<float> > &trainingData,
                 const std::vector<int> &trainingLabels,
                 const std::vector<std::vector<float>>   &validationData,
                 const std::vector<int> &validationLabels,
                 unsigned int max_iterations,
                 SkyNetDiagnostic           &diagnostic, SkynetTerminalInterface& exitter) = 0;

    virtual float getError( const std::vector< std::vector<float> > & data,  const std::vector<int> & labels)       = 0;
    virtual std::vector<int> & getClassification(const std::vector<std::vector<float>> & data)                                       = 0;
    virtual const std::string About() const                                                                             = 0;
    virtual void setWeights(std::vector< float > &all_weights)                                                          = 0;
    virtual void reshape(unsigned int num_inputs, unsigned int num_categories)                                          = 0;
    std::string Identify()
    {
        return std::string("ISkyNetClassificationProtocol");
    }
};
#endif //__SKYNET_PROTOCOL__

