#include "opencv2/opencv.hpp"
#include "ObjectDetector.h"
#include "ObjectdetectorFactory.h"
using namespace cv;
using namespace std;

typedef struct CallbackParams
{
	Mat input;
	Mat output;
	ObjectDetector* detector;
};

void paintCallback(int event, int x, int y, int flags, void* userdata)
{
	CallbackParams* params = (CallbackParams*)userdata;

	Mat input = params->input;
	Mat output = params->output;
	ObjectDetector* detector = params->detector;

	if (event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON)
	{
		//press left button and drag

		circle(input, Point(x, y), 1, Scalar(0,0,0), -1);
	}
	else if (event == EVENT_LBUTTONUP)
	{
		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;
		Mat mask = detector->findObjectsInImage(input, 90, 70, &objects, &numberOfObjects);
		output += mask;
	}
}



int main(int, char)
{
	Size size(600, 600);
	Mat drawingArea(size, CV_8UC3),
		outputImage(size, CV_8UC3);

	drawingArea = Scalar(255, 255, 255);
	outputImage = Scalar(0, 0, 0);

	Mat baseImage = imread("base.png");

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(8, 3);
	if (!detector->loadImage(baseImage))
		exit(1);

	CallbackParams* params = new CallbackParams;
	params->input = drawingArea;
	params->output = outputImage;
	params->detector = detector;

	namedWindow("Paint it!!", 1);
	namedWindow("Detected Result", 1);

	setMouseCallback("Paint it!!", paintCallback, params);

	for (;;)
	{
		imshow("Paint it!!", drawingArea);
		imshow("Detected Result", outputImage);

		if (waitKey(30) >= 0)
			break;
	}
	waitKey(0);
	return 0;
}