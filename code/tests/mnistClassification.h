#ifndef __MNIST_CLASSIFICATION__
#define __MNIST_CLASSIFICATION__
#include "tests.h"
#include <vector>
#include <protocol.h>
/*! Class provides an interface (implemetned) to work
 *  with MNIST database eg. read labels and images (train and test) 
*/

class mnistClassification
{
public:
    // Load and process MNIST database
    mnistClassification(std::string mnist_dirname);
    // Release resources
    ~mnistClassification();
    const std::vector<point> &getTrainingData();    //TODO: change returned type
    const std::vector<point> &getTestingData();
private:
    mnistClassification()
    {
    }
};


#endif //__MNIST_CLASSIFICATION__
