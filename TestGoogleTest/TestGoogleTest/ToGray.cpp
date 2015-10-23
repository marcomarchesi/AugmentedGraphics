#include "ToGray.h"

using namespace cv;

ToGray::ToGray()
{

}

cv::Mat ToGray::toGray(cv::Mat& input)
{
	if (input.channels() == 3)
	{
		Mat gray(input.size(), CV_8UC1);
		cvtColor(input, gray, CV_BGR2GRAY);
		return gray;
	}
	else if (input.channels() == 4)
	{
		Mat gray(input.size(), CV_8UC1);
		cvtColor(input, gray, CV_BGRA2GRAY);
		return gray;
	}
	else
		return input;
}