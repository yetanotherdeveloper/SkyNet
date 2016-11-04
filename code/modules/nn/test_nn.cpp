#include <cstdio>
#include <gtest/gtest.h>
#include <skynet.h>
#include "nn.h"
#include <memory>

// TODO: make fixture

/// Forward pass test
TEST(test_forward, test_nn)
{
  std::unique_ptr<NeuralNetwork> ip( 
          new NeuralNetwork(2, 2, GradientDescentType::STOCHASTIC, NeuronFlags::INIT_ONE));

  printf("Test Forward !\n");
}

// Gradient computation test
TEST(test_gradient, test_nn)
{
  printf("Test Gradient !\n");

  // See if Creating of Neural Net works

}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RUN_ALL_TESTS();
}
