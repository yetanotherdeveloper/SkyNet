#ifndef __MNIST_CLASSIFICATION__
#define __MNIST_CLASSIFICATION__
#include <vector>
#include <protocol.h>
/*! Class provides an interface (implemetned) to work
 *  with MNIST database eg. read labels and images (train and test) 
*/

class mnistClassification
{
    // Load and process MNIST database
    mnistClassification(std::string mnist_dirname);
    // Release resources
    ~mnistClassification();
    getTrainingData();
    getTestingData();
    
};

#endif //__MNIST_CLASSIFICATION__
