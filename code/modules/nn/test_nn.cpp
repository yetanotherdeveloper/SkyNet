#include <cstdio>
#include <gtest/gtest.h>
#include <skynet.h>

TEST(test_forward, test_nn)
{
  printf("Test Forward !\n");
}

TEST(test_gradient, test_nn)
{
  printf("Test Gradient !\n");
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  RUN_ALL_TESTS();
}
