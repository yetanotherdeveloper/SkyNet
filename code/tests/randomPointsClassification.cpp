#include <random>
#include "randomPointsClassification.h"
#include "os_inc.h"


/*!
 * \param  Generate
   !*/
randomPointsClassification::randomPointsClassification( unsigned int N )
{
    // Area from which random points are to be sampled
    m_minX = -1.0f;
    m_maxX = 1.0f;
    m_minY = -1.0f;
    m_maxY = 1.0f;

    makeRandomFunction();
    //makeFixedFunction( m_minX, 0.0f, m_maxX, 0.0f );

    // Generate learning data
    generateSet( m_trainingSet, N );
    // Generate data for testing generalization
    generateSet( m_testingSet, N );

    initWeights();
}


const std::vector< float > & randomPointsClassification::getInitialWeights()
{
    return m_iweights;
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

/// Validate input weights through training sets
unsigned int randomPointsClassification::validate()
{
    return getErrorRate( m_trainingSet, m_weights );
}


/*! Verification: calculate out of sample error eg. calculate error using
 * samples set not used for training
 */
unsigned int randomPointsClassification::verify()
{
    return getErrorRate( m_testingSet, m_weights );
}

/*!  Caluculate error of learning
 * by iterating through traiing set and comparing its learned classification with
 * the one used for learning. if they are diffrent then increase error rate
 */
unsigned int randomPointsClassification::getErrorRate( const std::vector< point > &samples,
                                                       const std::vector< float > &weights )
{
    unsigned int                         error_rate = 0;
    std::vector< point >::const_iterator it;

    if( weights.empty() != true )
    {

        for( it = samples.begin(); it != samples.end(); ++it )
        {
            if( classifyPoint( *it, weights ) != it->classification )
            {
                ++error_rate;
            }
        }
    }
    else
    {
        SKYNET_DEBUG( "Error: No weights were send from classification module -> setting maximal error\n" );
        error_rate = samples.size();
    }

    return error_rate;
}


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
        randPoint.classification = classifyPoint( randPoint, m_fweights );
        SKYNET_DEBUG( "point[%d]: x=%f y=%f class=%d\n", i, randPoint.x, randPoint.y, randPoint.classification );
        set.push_back( randPoint );
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

    SKYNET_DEBUG( "Fixed (target) w0=%f w1=%f w2=%f\n", m_fweights[0], m_fweights[1], m_fweights[2] );
}


/*! Idea is to take random two points on <min_x,rand_y> and <max_x,rand_y>
 */
void randomPointsClassification::makeRandomFunction()
{
    std::uniform_real_distribution< float > ud( 0.0f, 1.0f );
    std::random_device rd;

    // First random point generation <min_x,rand_y>
    float randval = ud( rd );
    float r1x     = m_minX;
    float r1y     = m_minY + ( m_maxY - m_minY ) * randval;

    // second random point generation <max_x,rand_y>
    randval = ud( rd );
    float r2x = m_maxX;
    float r2y = m_minY + ( m_maxY - m_minY ) * randval;

    // calculate A,B,C coefficients, where Ax + By + C = 0
    // y = (y_2 - y_1)/(x_2 - x_1)*(x - x_1) <=>
    // <=>  (x_2 - x_1)*y + (y_1 -y_2)*x + y_2*x_1 - y_1*x_1 = 0
    m_fweights.push_back( r2y * r1x - r1y * r1x );    //C
    m_fweights.push_back( r1y - r2y );                //A
    m_fweights.push_back( r2x - r1x );                //B

    SKYNET_DEBUG( "Generated (target) w0=%f w1=%f w2=%f\n", m_fweights[0], m_fweights[1], m_fweights[2] );
}

/*! Init weights
 */
void randomPointsClassification::initWeights()
{
    m_iweights.push_back( 0.0f );  //w0
    m_iweights.push_back( 0.0f );  //w1
    m_iweights.push_back( 0.0f );  //w2
    SKYNET_DEBUG( "Initial weights: w0=%f w1=%f w2=%f\n", m_iweights[0], m_iweights[1], m_iweights[2] );
}

