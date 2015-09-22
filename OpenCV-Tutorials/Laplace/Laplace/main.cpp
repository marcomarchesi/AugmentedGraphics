#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;

int main(int, char)
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	
	for (;;)
	{
		
		Mat src, src_gray, dst;
		int kernel_size = 3;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;
		const char* window_name = "Laplace Demo";
		cap >> src;
		if (src.empty())
		{
			return -1;
		}
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
		cvtColor(src, src_gray, COLOR_RGB2GRAY);
		namedWindow(window_name, WINDOW_AUTOSIZE);
		Mat abs_dst;
		Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(dst, abs_dst);
		imshow(window_name, abs_dst);

		if (waitKey(30) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}