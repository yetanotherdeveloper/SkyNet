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
        std::vector<float> m_fweights;      /// target function weights
        std::vector<float> m_weights;       /// hypotesis weights
        std::vector<float> m_iweights;       /// initial weights (weights to start with)
        std::vector<point> m_trainingSet;
        std::vector<point> m_testingSet;
    public:
        randomPointsClassification(unsigned int N);
        const std::vector<point> & getTrainingData();
        const std::vector<float> & getWeights();
        const std::vector<float> & getInitialWeights();
        void setWeights(const std::vector<float> &weights);
        unsigned int validate();
        unsigned int verify();
    private:
        unsigned int getErrorRate(const std::vector<point> &samples, const std::vector<float> &weights);
        void generateSet(std::vector<point> &set, unsigned int N);
        void makeRandomFunction();
        void initWeights();
        int classifyPoint(const point& sample, const std::vector<float> &weights);
};
#endif //__RANDOM_POINTS_CLASSIFFICATION__
