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
    const std::vector<std::vector<float>> &getTrainingData();
    const std::vector<std::vector<float>> &getValidationData();
    const std::vector<std::vector<float>> &getTestingData();
    const std::vector<int> &getTrainingLabels();
    const std::vector<int> &getValidationLabels();
    const std::vector<int> &getTestingLabels();

    void verify( const std::vector<int> & classification);
    void validate(const std::vector<int> & classification);

private:
    mnistClassification()
    {
    }

    // Load images and corressponding labels. Return number of items read (number of images has to match number of labels )
    unsigned int load_mnist(std::vector<std::vector<float>> &images,std::vector<int> &labels, std::string images_file, std::string labels_file);
    void showImage(const std::vector<float>& mnist_image);
    std::string getErrorRate(const std::vector<int> & classification, const std::vector<int> & expected_classification);

private:
    std::vector<std::vector<float>> m_train_images;
    std::vector<int> m_train_labels;
    unsigned int m_num_train_items;
    std::vector<std::vector<float>> m_test_images;
    std::vector<int> m_test_labels;
    unsigned int m_num_test_items;
};


#endif //__MNIST_CLASSIFICATION__
