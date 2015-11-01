#include "ObjectDetectorFactory.h"

using namespace cv;
using namespace std;
using namespace od;

int main(int, char)
{
	Mat baseImage = imread("mecha2.jpg");

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(58, 1);
	if(!detector->loadImage(baseImage))
		exit(1);

	/*
	Mat image = imread("OAC/chess.jpg");
	
	if (image.size().height > 800 || image.size().width > 800)
	{
		Size s = image.size(), small;
		small.height = s.height / 4.5;
		small.width = s.width / 4.5;

		resize(image, image, small);
	}

	
	vector<vector<vector<Point>>> objects;
	int numberOfObjects = 0;

	Mat mask = detector->findObjectsInImage(image, 50, 50, ObjectDetector::OutputMaskMode::CONVEX_HULL, &objects, &numberOfObjects);

	imshow("FINAL RESULT",mask);
	*/

	
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return 0;
	}


	Mat image;

	for (;;)
	{

		waitKey(0);
		cap >> image;
		

		if (image.size().height > 800 || image.size().width > 800)
		{
			Size s = image.size(), small;
			small.height = s.height / 3;
			small.width = s.width / 3;

			resize(image, image, small);
		}
		imshow("Source", image);
		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;

		

		Mat result = detector->findObjectsInImage(image, 80, 80, ObjectDetector::OutputMaskMode::CONVEX_HULL, &objects, &numberOfObjects);
		imshow("FINAL RESULT", result);

		if (waitKey(30) > 0)
			break;
	}
	

	waitKey(0);
	return 0;

}