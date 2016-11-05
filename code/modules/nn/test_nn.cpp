#include <cstdio>
#include <gtest/gtest.h>
#include <skynet.h>
#include "nn.h"
#include <memory>
#include <cmath>
#include <vector>

// TODO: make fixture
// TODO: make reference InnerProduct (as part of fixture)
// TODO: Generate some input data
// TODO: Run NeuralNetwork and reference and compare results
// TODO: Consider random weights (separate test apart from INIT_ONE and INIT_ZERO)
/// Forward pass test
TEST(test_forward, test_nn)
{
  // Generate data
  std::vector<float> input_data(5);
  input_data[0] = 0.0f;
  input_data[1] = 1.0f;
  input_data[2] = 2.0f;
  input_data[3] = 3.0f;
  input_data[4] = 4.0f;
  
  std::unique_ptr<NeuralNetwork> ip( 
          new NeuralNetwork(5, 2, GradientDescentType::STOCHASTIC, NeuronFlags::INIT_ONE));

  // Reference value to be (expected value) which is tanh of sum of tanh of input sum and bias(1.0f)
  float input_sum = input_data[0]*1.0f + input_data[1]*1.0f + input_data[2]*1.0f + input_data[3]*1.0f + 1.0f; 
  float ref_output = tanh((tanh(input_sum)) + (tanh(input_sum)) + 1.0f);
 
  // Get output from tested Network
  float computed_output = ip->getNetworkOutput(input_data); 

  EXPECT_FLOAT_EQ(ref_output, computed_output);
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
