#ifndef __RANDOM_POINTS_CLASSIFFICATION__
#define __RANDOM_POINTS_CLASSIFFICATION__
#include <vector>
/*! Object represting an instance of random point classification class
 *  that aims to deliver input, output data , verification and other
 *  tools to condact classification test of random data
 *
   !*/
class randomPointsClassification
{
    private:
        struct point {
            float x;                //! first coord of point
            float y;                //! second coord of point
            int classification;     //! +1 or -1 meaning the classification area
        };
        struct weights
        {
            float w0;
            float w1;
            float w2;
        };
        float              m_minX, m_maxX;
        float              m_minY, m_maxY;
        float              m_A, m_B, m_C; //! Random line equation coefficients
        weights            m_weights; 
        std::vector<point> m_trainingSet;
    public:
        randomPointsClassification(unsigned int N);
    private:
        void makeRandomFunction();
        void initWeights();
};
#endif //__RANDOM_POINTS_CLASSIFFICATION__
