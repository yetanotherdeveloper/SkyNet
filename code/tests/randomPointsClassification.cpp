#include <random>
#include "randomPointsClassification.h"
#include "os_inc.h"


/*!
 * \param  Generate
   !*/
randomPointsClassification::randomPointsClassification(unsigned int N)
{
    // Area from which random points are to be sampled
    m_minX = -1.0f;
    m_maxX = 1.0f;
    m_minY = -1.0f;
    m_maxY = 1.0f;

    makeRandomFunction();

    // Generate learning data
    generateSet(m_trainingSet,N);
    // Generate data for testing generalization
    generateSet(m_testingSet,N);

    initWeights();
}


const std::vector<point> & randomPointsClassification::getTrainingData()
{
    return m_trainingSet;
}


void randomPointsClassification::generateSet(std::vector<point> &set, unsigned int N)
{
    set.clear();

    // Get N random points
    std::uniform_real_distribution<float> randx(m_minX,m_maxX);
    std::uniform_real_distribution<float> randy(m_minY,m_maxY);
    std::random_device rd;

    point randPoint;
    for(int i = 0; i < N; ++i)
    {
        // get point
        randPoint.x = randx(rd);
        randPoint.y = randy(rd);
        // get its classification value eg.
        // sign(A*x + B*y +C) where x,y are above chosen by random points
        randPoint.classification = (m_A * randPoint.x + m_B * randPoint.y + m_C >= 0.0f) ? 1 : -1;
        SKYNET_DEBUG("point[%d]: x=%f y=%f class=%d\n",i,randPoint.x,randPoint.y,randPoint.classification);
        set.push_back(randPoint);
    }

}

/*! Idea is to take random two points on <min_x,rand_y> and <max_x,rand_y>
 */
void randomPointsClassification::makeRandomFunction()
{
    std::uniform_real_distribution<float> ud(0.0f,1.0f);
    std::random_device rd;

    // First random point generation <min_x,rand_y>
    float randval = ud(rd);
    float r1x     = m_minX;
    float r1y     = m_minY + (m_maxY - m_minY) * randval;

    // second random point generation <max_x,rand_y>
    randval = ud(rd);
    float r2x = m_maxX;
    float r2y = m_minY + (m_maxY - m_minY) * randval;

    // calculate A,B,C coefficients, where Ax + By + C = 0
    // y = (y_2 - y_1)/(x_2 - x_1)*(x - x_1) <=>
    // <=>  (x_2 - x_1)*y + (y_1 -y_2)*x + y_2*x_1 - y_1*x_1 = 0
    m_A = r1y - r2y;
    m_B = r2x - r1x;
    m_C = r2y * r1x - r1y * r1x;

    SKYNET_DEBUG("Generated A=%f B=%f C=%f\n",m_A,m_B,m_C);
}

/*! Init weights
 */
void randomPointsClassification::initWeights()
{
    m_weights.push_back(0.0f);  //w0
    m_weights.push_back(0.0f);  //w1
    m_weights.push_back(0.0f);  //w2
    SKYNET_DEBUG("Initial weights: w0=%f w1=%f w2=%f\n",m_weights[0],m_weights[1],m_weights[2]);
}
