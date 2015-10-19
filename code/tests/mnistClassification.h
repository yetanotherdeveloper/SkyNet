#ifndef __MNIST_CLASSIFICATION__
#define __MNIST_CLASSIFICATION__
#include "tests.h"
#include <vector>
#include <memory>
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
    ~mnistClassification() {};
    const std::vector<point> &getTrainingData();    //TODO: change returned type
    const std::vector<point> &getTestingData();
private:
    mnistClassification()
    {
    }

    // Load images and corressponding labels. Return number of items read (number of images has to match number of labels )
    unsigned int load_mnist(std::vector<std::unique_ptr<float>> &images,std::vector<char> &labels, std::string images_file, std::string labels_file);
    void showImage(const float* mnist_image);

private:
    std::vector<std::unique_ptr<float>> m_train_images;
    std::vector<char> m_train_labels;
    unsigned int m_num_train_items;
    std::vector<std::unique_ptr<float>> m_test_images;
    std::vector<char> m_test_labels;
    unsigned int m_num_test_items;
};


#endif //__MNIST_CLASSIFICATION__
