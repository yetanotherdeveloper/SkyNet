#include "mnistClassification.h"

#include <iostream>
#include <istream>
#include <vector>
#include <fstream>
#include <Magick++.h>
#include <memory>
#include <cassert>

unsigned int  mnistClassification::load_mnist(std::vector<std::unique_ptr<float>> &images,std::vector<char> &labels, std::string images_file, std::string labels_file)
{
    images.clear();
    labels.clear();

    std::ifstream ifs(labels_file.c_str(), std::ifstream::binary); 
    if(ifs.good() == false)
    {
        throw std::runtime_error("""Error opening MNIST labels file."
                                    "please use --mnist-data to specify"
                                    "directory with mnist data");
    } 

    auto block_to_int = [](unsigned char* block) 
    {
        return ((((((u_int32_t)block[0] << 8) + (u_int32_t)block[1]) << 8)  + (u_int32_t)block[2]) << 8 ) + (u_int32_t)block[3];
    };

    std::cout << "Labels:" << std::endl;
    unsigned char int_read[sizeof(u_int32_t)]; 
    ifs.read((char*)int_read,sizeof(u_int32_t));
    u_int32_t magic_number = block_to_int(int_read); 
    std::cout << "  Magic Number: " << magic_number << std::endl;

    ifs.read((char*)int_read,sizeof(u_int32_t));
    u_int32_t num_items = block_to_int(int_read); 
    std::cout << "  Number of Items: " << num_items << std::endl;

    // Make vecotr for labels equal to size of number of labels to be hold
    labels.resize(num_items);    
    for(u_int32_t i=0; i<num_items; ++i) 
    {
        ifs.read(&labels[i],sizeof(char));
    }
    ifs.close();

    std::ifstream iifs(images_file.c_str(), std::ifstream::binary); 
    if(iifs.good() == false)
    {
        throw std::runtime_error("Error opening MNIST images file");
    } 
 
    std::cout << "Images:" << std::endl;
    iifs.read((char*)int_read,sizeof(u_int32_t));
    magic_number = block_to_int(int_read); 
    std::cout << "  Magic Number: " << magic_number << std::endl;
    iifs.read((char*)int_read,sizeof(u_int32_t));
    u_int32_t num_images = block_to_int(int_read); 
    std::cout << "  Number of Images: " << num_images << std::endl;
    iifs.read((char*)int_read,sizeof(u_int32_t));
    u_int32_t num_rows = block_to_int(int_read); 
    std::cout << "  Number of Rows: " << num_rows << std::endl;
    iifs.read((char*)int_read,sizeof(u_int32_t));
    u_int32_t num_cols = block_to_int(int_read); 
    std::cout << "  Number of Cols: " << num_cols << std::endl;

    if( num_items != num_images)
    {
        throw std::runtime_error(" Error number of read labels does not match number of read images.");
    }

    std::unique_ptr<unsigned char> single_image_raw(new unsigned char[num_rows*num_cols]);

    for(unsigned int i=0; i<num_items;++i)
    {
        iifs.read((char*)single_image_raw.get(),sizeof(char)*num_cols*num_rows);        
        // convert image to coords of read images (width and height)
        images.push_back(std::unique_ptr<float>(new float[num_rows*num_cols]));
        for(u_int32_t pix = 0; pix< num_cols*num_rows;++pix)
        {
            *(images[i].get() + pix) = ((unsigned char)*(single_image_raw.get() +pix))* 1.0f/255.0f; 
        }
            
    }
    iifs.close();

    return num_items;
}

void mnistClassification::showImage(const float* mnist_image)
{
    Magick::Image test_image(Magick::Geometry(28,28),"black"); 

    using namespace Magick;
    for(u_int32_t y=0; y< 28; ++y)
    {
        for(u_int32_t x=0; x< 28; ++x)
        {
                u_int32_t i = y*28+x;
                float bum = *(mnist_image + i)*QuantumRange;
                test_image.pixelColor(x,y,Magick::Color(bum,bum,bum));
        }
    }

    test_image.display();
}

mnistClassification::mnistClassification(std::string mnist_dirname)
{
    m_num_train_items = load_mnist(m_train_images,m_train_labels,mnist_dirname+"/train-images-idx3-ubyte",mnist_dirname+"/train-labels-idx1-ubyte");

    m_num_test_items = load_mnist(m_test_images,m_test_labels,mnist_dirname+"/train-images-idx3-ubyte",mnist_dirname+"/train-labels-idx1-ubyte");

    // diagnostic code to draw mnist image
    //showImage(m_train_images[0].get());
}

namespace {

void runTest(SkyNet& skynet_instance)
{
    std::cout << "MNIST Test executing!" << std::endl;

    mnistClassification mnist_test(skynet_instance.getMnistDataDir());
    SkyNetDiagnostic diagnostic;
    auto classifiers = skynet_instance.getClassificationModules();
    SkynetTerminalInterface                     exitter('q');

    for(auto& classifier : classifiers) 
    {
        //TODO: execution of mnist Test    
    }
}
auto a = TestsRegistry::inst().addTest("MNIST",runTest);
}