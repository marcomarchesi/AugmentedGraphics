#include "ObjectDetectorFactory.h"

using namespace cv;

int main(int, char)
{
	Mat baseImage = imread("ninja.jpg");

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(8, 2);
	detector->loadImage(baseImage);

	/*
	Mat image = imread("stelle.jpg");
	Mat result = detector->findObjectsInImage(image, 70.0, 70.0);
	imshow("FINAL RESULT", result);
	*/

	
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return 0;
	}

	Mat image;

	for (;;)
	{
		cap >> image;

		Mat result = detector->findObjectsInImage(image, 85, 85);
		imshow("FINAL RESULT", result);

		if (waitKey(30) > 0)
			break;
	}
	

	waitKey(0);
	return 0;

}