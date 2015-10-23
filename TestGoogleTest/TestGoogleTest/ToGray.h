#include "opencv2/opencv.hpp"



class ToGray
{
public:

	ToGray();
	cv::Mat toGray(cv::Mat& input);
};