#include "ObjectDetectorFactory.h"
#include <time.h>

using namespace cv;
using namespace std;
using namespace od;

int main(int, char)
{
	Mat baseImage = imread("mecha2.jpg");

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(52, 1);
	if(!detector->loadImage(baseImage))
		exit(1);

	/*
	Mat image = imread("mecha2.jpg");

	if (image.empty())
		exit(2);
	
	
	if (image.size().height > 800 || image.size().width > 800)
	{
		Size s = image.size(), small;
		small.height = s.height / 3;
		small.width = s.width / 3;

		resize(image, image, small);
	}
	
	
	vector<vector<vector<Point>>> objects;
	int numberOfObjects = 0;

	Mat mask = detector->findObjectsInImage(image, 50, 50, ObjectDetector::OutputMaskMode::CONVEX_HULL, &objects, &numberOfObjects);

	imshow("FINAL RESULT",mask);
	*/

	
	Mat image;
	Mat fpsImg(Size(100, 50), CV_8UC1);

	time_t start, end;
	int counter = 0;

	VideoCapture cap(1);
	if (!cap.isOpened())
	{
		return 0;
	}

	/*
	time(&start);
	for (;;)
	{
		//waitKey(0);
		cap >> image;

		imshow("Source", image);


		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;
		time(&end);

		counter++;
		double sec = difftime(end, start);
		double fps = counter / sec;

		fpsImg = Scalar(255);
		putText(fpsImg, to_string(fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0));
		imshow("FPS", fpsImg);
		
		if (waitKey(30) > 0)
			break;
	}
	*/
	
	counter = 0;
	time(&start);	

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
		imshow("Source", image);

		
		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;
		
		

		Mat result = detector->findObjectsInImage(image, 75, 50, ObjectDetector::OutputMaskMode::CONVEX_HULL, &objects, &numberOfObjects);
		
		
		time(&end);

		counter++;
		double sec = difftime(end, start);
		double fps = counter / sec;

		fpsImg = Scalar(255);
		putText(fpsImg, to_string(fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0));
		imshow("FPS", fpsImg);

		imshow("FINAL RESULT", result);

		if (waitKey(30) > 0)
			break;
	}
	

	waitKey(0);
	return 0;

}