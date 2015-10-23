#include "opencv2/opencv.hpp"
#include <gtest\gtest.h>
#include "ToGray.h"

using namespace cv;

TEST(TestToGray, work)
{
	Mat input = imread("input.jpg");
	ASSERT_FALSE(input.empty());
	
	ToGray tg;

	Mat gray = tg.toGray(input);
	EXPECT_EQ(1, gray.channels());
}




int main(int argc, char* argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	 int ret = RUN_ALL_TESTS();

	 getchar();
	 return ret;
}