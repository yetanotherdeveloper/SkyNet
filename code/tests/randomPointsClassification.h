#ifndef __RANDOM_POINTS_CLASSIFFICATION__
#define __RANDOM_POINTS_CLASSIFFICATION__
#include "tests.h"
#include <vector>
#include <protocol.h>
/*! Object represting an instance of random point classification class
 *  that aims to deliver input, output data , verification and other
 *  tools to condact classification test of random data
 *
   !*/

class randomPointsClassification 
{
    private:
        float              m_minX, m_maxX;
        float              m_minY, m_maxY;
        std::vector<float> m_fweights;      /// target function weights
        std::vector<float> m_weights;       /// hypotesis weights
        std::vector<float> m_iweights;       /// initial weights (weights to start with)
        std::vector<std::vector<float>> m_trainingData;
        std::vector<int> m_trainingLabels;
        std::vector<std::vector<float>> m_validationData;
        std::vector<int> m_validationLabels;
        std::vector<std::vector<float>> m_testingData;
        std::vector<int> m_testingLabels;
    public:
        randomPointsClassification(unsigned int N, unsigned int nrLines);
        const std::vector<std::vector<float>> & getTrainingData();
        const std::vector<int> & getTrainingLabels();
        const std::vector< std::vector<float> > & getValidationData();
        const std::vector< int > & getValidationLabels();
        const std::vector<std::vector<float>> & getTestingData();
        const std::vector<int> & getTestingLabels();
        const std::vector<float> & getWeights();
		const std::vector<float> & getTargetWeights();
        void setWeights(const std::vector<float> &weights);
        void validate(const std::vector<int> & classification);
        void verify( const std::vector<int> & classification);
        unsigned int validate();
        unsigned int verify();
    private:
        std::string getErrorRate(const std::vector<int> & classification, const std::vector<int> & expected_classification);
        void generateSet( std::vector< std::vector<float> > &data, std::vector<int> &labels , unsigned int N );
        void makeRandomFunctions(unsigned int nrLines);
        void makeFixedFunction( float x1, float y1, float x2, float y2 );
//        int classifyPoint(const std::vector<float>& sample, const std::vector<float> &weights);
        int classifyPoint(const std::vector<float>& sample);
};


#endif //__RANDOM_POINTS_CLASSIFFICATION__
