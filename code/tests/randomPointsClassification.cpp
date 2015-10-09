#include <random>
#include <cassert>
#include "randomPointsClassification.h"
#include "mnistClassification.h"
#include "tests.h"
#include "os_inc.h"


/*!
 * \param  Generate
   !*/
randomPointsClassification::randomPointsClassification(unsigned int N, unsigned int nrLines)
{
    // Area from which random points are to be sampled
    m_minX = -1.0f;
    m_maxX = 1.0f;
    m_minY = -1.0f;
    m_maxY = 1.0f;

    makeRandomFunctions(nrLines); 
    //makeFixedFunction( -1.0f, 0.0f, 1.0f, -2.0f ); // temporar diagnostic solution

    // Generate learning data
    generateSet( m_trainingSet, N );
    // Generate validation set (used for early stopping)
    generateSet( m_validationSet, N/2 );
    // Generate data for testing generalization
    generateSet( m_testingSet, N );
}

const std::vector<float> & randomPointsClassification::getTargetWeights()
{
    return m_fweights;
}

const std::vector< float > & randomPointsClassification::getWeights()
{
    return m_weights;
}


void randomPointsClassification::setWeights( const std::vector< float > &weights )
{
    m_weights = weights;
}

/// Validate input classification data through training sets
float randomPointsClassification::validate(const std::vector<int> & classification)
{
    return getErrorRate( m_trainingSet, classification );
}


/*! Verification: calculate out of sample error eg. calculate error using
 * samples set not used for training
 */
float randomPointsClassification::verify( const std::vector<int> & classification)
{
    return getErrorRate( m_testingSet, classification );
}

/*!  Caluculate error of learning
 * by iterating through traiing set and comparing its learned classification with
 * the one used for learning. if they are diffrent then increase error rate
 */
float randomPointsClassification::getErrorRate(const std::vector<point> &samples,  const std::vector<int> & classification)
{
    unsigned int                         error_rate = 0;
    std::vector< point >::const_iterator it;

    // If number of samples does not corresspond to number of classification data then
    // something is very wrong and we return max error rate: 1.0
    if( classification.size() != samples.size())
    {
        assert(classification.size() == samples.size());
        SKYNET_INFO("Error: classification data size and samples data size do differ!\n");
        return 1.0f; 
    }

    for(unsigned int i =0; i< samples.size(); ++i) 
    {
        if(  classification[i]  != samples[i].classification)
        {
            ++error_rate;
        }
    }
    
    return error_rate/(float)samples.size();
}
///////////////////////////////////////////////////////////////////////////
const std::vector< point > & randomPointsClassification::getValidationData()
{
    return m_validationSet;
}
////////////////////////////////////////////////////////////////////////////
const std::vector< point > & randomPointsClassification::getTrainingData()
{
    return m_trainingSet;
}


const std::vector<point> & randomPointsClassification::getTestingData()
{
    return m_testingSet;
}


/// sign(A*x + B*y +C) where x,y are above chosen by random points
int randomPointsClassification::classifyPoint( const point& sample, const std::vector< float > &weights )
{
    return ( weights[1] * sample.x + weights[2] * sample.y + weights[0] >= 0.0f ) ? 1 : -1;
}

/*! there is a number of separating lines. We classify given point
 *  as +1 class if multiplication  of all (An*x + Bn*y +Cn)
 *  where x,y are above chosen by random points, for n <1..nrLines>
 *  is negative. This should give as intresting set
 */
int randomPointsClassification::classifyPoint(const point& sample)
{
    float result = 1.0f;

    for(unsigned int i = 0; i < m_fweights.size(); i += 3)
    {
        result *= (m_fweights[i+1] * sample.x + m_fweights[i+2] * sample.y + m_fweights[i]);
    }
    return result < 0.0f ? -1 : 1;  //temporary hack
}

void randomPointsClassification::generateSet( std::vector< point > &set, unsigned int N )
{
    set.clear();

    // Get N random points
    std::uniform_real_distribution< float > randx( m_minX, m_maxX );
    std::uniform_real_distribution< float > randy( m_minY, m_maxY );
    std::random_device rd;

    point randPoint;
    for( int i = 0; i < N; ++i )
    {
        // get point
        randPoint.x = randx( rd );
        randPoint.y = randy( rd );
        // get its classification value eg.
        randPoint.classification = classifyPoint(randPoint);
        //SKYNET_DEBUG("point[%d]: x=%f y=%f class=%d\n",i,randPoint.x,randPoint.y,randPoint.classification);
        set.push_back(randPoint);
    }

}


/*! Calculate line coefficients (weights) for given two points
 *  eg. get line coefficients so that this line go through given two points
 *  This is mainly for diagnostic
 */
void randomPointsClassification::makeFixedFunction( float x1, float y1, float x2, float y2 )
{
    // calculate A,B,C coefficients, where Ax + By + C = 0
    // y = (y_2 - y_1)/(x_2 - x_1)*(x - x_1) <=>
    // <=>  (x_2 - x_1)*y + (y_1 -y_2)*x + y_2*x_1 - y_1*x_1 = 0
    m_fweights.push_back( y2 * x1 - y1 * x1 );    //C
    m_fweights.push_back( y1 - y2 );              //A
    m_fweights.push_back( x2 - x1 );              //B

    //SKYNET_DEBUG( "Fixed (target) w0=%f w1=%f w2=%f\n", m_fweights[0], m_fweights[1], m_fweights[2] );
}

/*! We generate given number of separation lines acording to following idea: 
 * taking  random two points on <min_x,rand_y> and <max_x,rand_y>
 *  
 */
void randomPointsClassification::makeRandomFunctions(unsigned int nrLines)
{
    std::uniform_real_distribution<float> ud(0.0f,1.0f);
    std::random_device rd;

    for(unsigned int i = 0; i < nrLines; ++i)
    {
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
        m_fweights.push_back(r2y * r1x - r1y * r1x); //C
        m_fweights.push_back(r1y - r2y);            //A
        m_fweights.push_back(r2x - r1x);            //B
    }
}



namespace {
void runTest(void)
{
    std::cout << "Random points Test executing!" << std::endl;
}
auto a = TestsRegistry::inst().addTest("Random points", runTest);
}
