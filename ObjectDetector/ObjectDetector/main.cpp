#include "ObjectDetectorFactory.h"

using namespace cv;
using namespace std;
using namespace od;

int main(int, char)
{
	Mat baseImage = imread("dragonTower1.jpg");

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(31, 1);
	if(!detector->loadImage(baseImage))
		exit(1);

	/*
	Mat image = imread("car.jpg");
	
	if (image.size().height > 800 || image.size().width > 800)
	{
		Size s = image.size(), small;
		small.height = s.height / 3;
		small.width = s.width / 3;

		resize(image, image, small);
	}

	
	vector<vector<vector<Point>>> objects;
	int numberOfObjects = 0;

	Mat mask = detector->findObjectsInImage(image, 88, 90, &objects, &numberOfObjects);

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

		//waitKey(0);
		cap >> image;
		

		if (image.size().height > 800 || image.size().width > 800)
		{
			Size s = image.size(), small;
			small.height = s.height / 3;
			small.width = s.width / 3;

			resize(image, image, small);
		}
		//imshow("Source", image);
		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;

		

		Mat result = detector->findObjectsInImage(image, 70, 90, ObjectDetector::OutputMaskMode::PRECISE, &objects, &numberOfObjects);
		imshow("FINAL RESULT", result);

		if (waitKey(30) > 0)
			break;
	}
	

	waitKey(0);
	return 0;

}