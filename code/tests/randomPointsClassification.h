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
        std::vector<point> m_trainingSet;
        std::vector<point> m_validationSet;
        std::vector<point> m_testingSet;
    public:
        randomPointsClassification(unsigned int N, unsigned int nrLines);
        const std::vector<point> & getTrainingData();
        const std::vector< point > & getValidationData();
        const std::vector<point> & getTestingData();
        const std::vector<float> & getWeights();
		const std::vector<float> & getTargetWeights();
        void setWeights(const std::vector<float> &weights);
        float validate(const std::vector<int> & classification);
        float verify(const std::vector<int> & classification);
        unsigned int validate();
        unsigned int verify();
    private:
        float getErrorRate(const std::vector<point> &samples, const std::vector<int> & classification);
        void generateSet(std::vector<point> &set, unsigned int N);
        void makeRandomFunctions(unsigned int nrLines);
        void makeFixedFunction( float x1, float y1, float x2, float y2 );
        int classifyPoint(const point& sample, const std::vector<float> &weights);
        int classifyPoint(const point& sample);
};


#endif //__RANDOM_POINTS_CLASSIFFICATION__
