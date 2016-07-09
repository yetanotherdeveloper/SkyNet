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
    generateSet( m_trainingData, m_trainingLabels, N );
    // Generate validation set (used for early stopping)
    generateSet( m_validationData, m_validationLabels, N/2 );
    // Generate data for testing generalization
    generateSet( m_testingData, m_testingLabels, N );
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
void randomPointsClassification::validate(const std::vector<int> & classification)
{
    std::cout << "Training error: " + getErrorRate( classification, m_trainingLabels ) << std::endl;
}


/*! Verification: calculate out of sample error eg. calculate error using
 * samples set not used for training
 */
void randomPointsClassification::verify( const std::vector<int> & classification)
{
    std::cout << "Testing error: " +getErrorRate( classification, m_testingLabels ) << std::endl;
}

/*!  Caluculate error of learning
 * by iterating through traiing set and comparing its learned classification with
 * the one used for learning. if they are diffrent then increase error rate
 */
std::string randomPointsClassification::getErrorRate(const std::vector<int> & classification, const std::vector<int> & expected_classification)
{
    unsigned int                         error_rate = 0;
    std::vector< std::vector<float> >::const_iterator it;

    // If number of samples does not corresspond to number of classification data then
    // something is very wrong and we return max error rate: 1.0
    if( classification.size() != expected_classification.size())
    {
        assert(classification.size() == expected_classification.size());
        SKYNET_INFO("Error: classification data size and expected classification data size do differ!\n");
        return  std::to_string(expected_classification.size()) + "out of " + std::to_string(expected_classification.size());
    }

    for(unsigned int i =0; i< classification.size(); ++i) 
    {
        if( classification[i]  != expected_classification[i])
        {
            ++error_rate;
        }
    }
    
    return std::to_string(error_rate) + "out of " + std::to_string(expected_classification.size());
}
///////////////////////////////////////////////////////////////////////////
const std::vector< std::vector<float> > & randomPointsClassification::getValidationData()
{
    return m_validationData;
}
////////////////////////////////////////////////////////////////////////////
const std::vector< int > & randomPointsClassification::getValidationLabels()
{
    return m_validationLabels;
}
////////////////////////////////////////////////////////////////////////////
const std::vector< std::vector<float> > & randomPointsClassification::getTrainingData()
{
    return m_trainingData;
}
////////////////////////////////////////////////////////////////////////////
const std::vector<int> & randomPointsClassification::getTrainingLabels()
{
    return m_trainingLabels;
}
////////////////////////////////////////////////////////////////////////////

const std::vector<std::vector<float>> & randomPointsClassification::getTestingData()
{
    return m_testingData;
}
////////////////////////////////////////////////////////////////////////////
const std::vector<int> & randomPointsClassification::getTestingLabels()
{
    return m_testingLabels;
}
////////////////////////////////////////////////////////////////////////////

/*! there is a number of separating lines. We classify given point
 *  as +1 class if multiplication  of all (An*x + Bn*y +Cn)
 *  where x,y are above chosen by random points, for n <1..nrLines>
 *  is negative. This should give as intresting set
 */
int randomPointsClassification::classifyPoint(const std::vector<float>& sample)
{
    float result = 1.0f;

    for(unsigned int i = 0; i < m_fweights.size(); i += 3)
    {
        result *= (m_fweights[i+1] * sample[0] + m_fweights[i+2] * sample[1] + m_fweights[i]);
    }
    return result < 0.0f ? -1 : 1;  //temporary hack
}

void randomPointsClassification::generateSet( std::vector< std::vector<float> > &data, std::vector<int> &labels , unsigned int N )
{
    data.clear();
    data.resize(N);
    labels.clear();

    // Get N random points
    std::uniform_real_distribution< float > randx( m_minX, m_maxX );
    std::uniform_real_distribution< float > randy( m_minY, m_maxY );
    std::random_device rd;

    std::vector<float> randPoint;
    for( int i = 0; i < N; ++i )
    {
        // get point
        data[i].push_back(randx( rd ));
        data[i].push_back(randy( rd ));
        // get its classification value eg.
        labels.push_back(classifyPoint(data[i]));
        //SKYNET_DEBUG("point[%d]: x=%f y=%f class=%d\n",i,randPoint.x,randPoint.y,randPoint.classification);
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
void runTest(SkyNet& skynet_instance)
{
    std::cout << "Random points Test executing!" << std::endl;
    SkyNetDiagnostic diagnostic;
    randomPointsClassification rpc(100,2);
    SkynetTerminalInterface                     exitter('q');

    // diagnostic results are stored in directory named after process ID
    auto classifiers = skynet_instance.getClassificationModules();

    for(auto& classifier : classifiers) 
    {
        diagnostic.reset();
        SKYNET_INFO("Running Random points classification test against: %s\n",classifier.module->About().c_str() );
        // Pass Input data , and initial weights to RunCL , RunRef functions
        classifier.module->reshape(2,2);    // Input is two dimensional (x,y) , and two categories are to be chosen from (eg. target values)
        classifier.module->RunRef(rpc.getTrainingData(), 
                                  rpc.getTrainingLabels(),
                                  rpc.getValidationData(),
                                  rpc.getValidationLabels(), 
                                  skynet_instance.getMaxIterations(),
                                  diagnostic,
                                  exitter );
        SKYNET_INFO("GetError: %f\n",classifier.module->getError(rpc.getTrainingData(), rpc.getTrainingLabels() ) );
        rpc.validate(classifier.module->getClassification(rpc.getTrainingData()));
        rpc.verify(classifier.module->getClassification(rpc.getTestingData()));
        diagnostic.makeWeightsAnalysis(classifier.module->About());
        diagnostic.saveWeightsToFile(classifier.module->About());
        diagnostic.makeTrainingAnalysis(classifier.module->About(),
                                        rpc.getTrainingData(),
                                        rpc.getTrainingLabels(), 
                                        rpc.getTargetWeights(),
                                        rpc.getWeights() );
        diagnostic.makeGeneralizationAnalysis(classifier.module->About(),
                                              rpc.getTestingData(), 
                                              rpc.getTestingLabels(),
                                              rpc.getTargetWeights(),
                                              rpc.getWeights() );
    }
}
        int classifyPoint(const std::vector<float>& sample, const std::vector<float> &weights);
auto a = TestsRegistry::inst().addTest("Random points", runTest);
}
