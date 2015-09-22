#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


int main(int, char)
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) // check if we succeeded
		return -1;
	
	for (;;)
	{
		Mat src, dst;
		const char* source_window = "Source image";
		const char* equalized_window = "Equalized Image";

		cap >> src;

		if (src.empty())
		{
			cout << "Usage: ./Histogram_Demo <path_to_image>" << endl;
			return -1;
		}
		cvtColor(src, src, COLOR_BGR2GRAY);

		equalizeHist(src, dst);

		namedWindow(source_window, WINDOW_AUTOSIZE);
		namedWindow(equalized_window, WINDOW_AUTOSIZE);
		imshow(source_window, src);
		imshow(equalized_window, dst);

		if (waitKey(30) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}