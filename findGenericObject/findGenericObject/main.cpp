#include "GenericObject.h"

using namespace cv;
using namespace std;




int main(int, char)
{
	Mat baseImage = imread("ninja.jpg");

	GenericObject generic(baseImage);


	// READ IMAGE

	Mat image = imread("stella3.jpg");
	for (;;)
	{
		Mat result = generic.findObjectsInImg(image, 92.0);
		imshow("FINAL RESULT", result);
		waitKey(0);
	}



	// VIDEO CAPTURE
	/*
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
	return 0;
	}

	Mat image;

	for (;;)
	{
	cap >> image;

	Mat result = generic.findObjectsInImg(image, 92.0);
	imshow("FINAL RESULT", result);

	if (waitKey(30) > 0)
	break;
	}
	*/


	waitKey(0);
	return 0;
}