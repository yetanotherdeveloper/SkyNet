#include <cstdio>
#include <gtest/gtest.h>
#include <skynet.h>

TEST(test_max_iterations, test_commandline)
{
  unsigned int max_iterations = 123;
  std::string command_line("--max_iterations=");
  command_line.append(std::to_string(max_iterations));

  const char* c_strings[2];
  c_strings[0] = "./test_commandline"; 
  c_strings[1] = (command_line.c_str());
  SkyNet m_skynet(2,(char* const*)c_strings);

  EXPECT_EQ(max_iterations,m_skynet.getMaxIterations());
}


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
	printf("Hello ULT World!\n");
  RUN_ALL_TESTS();
}
