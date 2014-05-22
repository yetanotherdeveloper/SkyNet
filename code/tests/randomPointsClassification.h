#ifndef __RANDOM_POINTS_CLASSIFFICATION__
#define __RANDOM_POINTS_CLASSIFFICATION__
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
        float              m_A, m_B, m_C;     //! Random line equation coefficients
        std::vector<float> m_weights;
        std::vector<point> m_trainingSet;
        std::vector<point> m_testingSet;
    public:
        randomPointsClassification(unsigned int N);
        const std::vector<point> & getTrainingData();
        const std::vector<point> & getWeights();
    private:
        void generateSet(std::vector<point> &set, unsigned int N);
        void makeRandomFunction();
        void initWeights();
};
#endif //__RANDOM_POINTS_CLASSIFFICATION__
