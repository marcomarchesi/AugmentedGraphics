#include "MultiContoursObject.h"

using namespace cv;
using namespace std;




int main(int, char)
{
	Mat baseImage = imread("base.png");

	MultiContoursObject multi(baseImage, 4);


	// READ IMAGE

	Mat image = imread("mess.png");
	if (image.empty())
		exit(1);

	/*
	every time you press a button on the keyboard
	you expand the active area of ​​the image,
	when an object is detected the resize is locked
	*/

	/*
	Comment/uncomment the DEBUG_MODE in commonInclude.h
	to hide/show the debug data and image
	*/

	
	//for (;;)
	//{
		Mat result = multi.findObjectsInImg(image, 20.0, 20.0);
		imshow("FINAL RESULT", result);
		waitKey(0);
	//}
	


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