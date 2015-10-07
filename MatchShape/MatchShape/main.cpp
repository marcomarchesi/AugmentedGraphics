
#include "Star.h"

using namespace cv;
using namespace std;

void findAndApproxContours(Mat& frame, vector<vector<Point>>& retContours, bool findStar)
{
	Mat thresh(frame.size(), CV_8UC1);
	cvtColor(frame, thresh, CV_BGR2GRAY);
	int soglia;

	if (findStar)
		soglia = 127;
	else
		soglia = 70;

	threshold(thresh, thresh, soglia, 255, THRESH_BINARY);

	//imshow("threshold", thresh);

	vector<vector<Point>> contours;
	vector<Point> approx;
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	Mat contoursImage(thresh.size(), CV_8UC1);
	contoursImage = cv::Scalar(0);
	drawContours(contoursImage, contours, -1, cv::Scalar(255), 1, CV_AA);

	imshow("Contours", contoursImage);

	retContours.clear();

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 5)
			continue;

		double epsilon = contours[i].size() * 0.05;
		approxPolyDP(contours[i], approx, epsilon, true);

		if (approx.size() != 8 && findStar)
			continue;

		retContours.push_back(approx);
	}
}



int main(int, char)
{
	Mat baseImage = imread("ninja.jpg");

	Star star(baseImage);
		

	// TEST matchShapes
	/*
	cout << "TEST MatchShapes" << endl;

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 5)
			continue;

		double hamming = matchShapes(contours[i], approxC[0], CV_CONTOURS_MATCH_I1, 0.0);
		cout << "Contours " + to_string(i) << " --> " << hamming << endl;

		Mat temp(thresh.size(), CV_8UC1);
		temp = cv::Scalar(0);
		vector<vector<Point>> tempVect;
		tempVect.push_back(contours[i]);
		drawContours(temp, tempVect, -1, cv::Scalar(255), 1, CV_AA);
		imshow("Contours " + to_string(i), temp);
	}
	*/
	//double hamming = matchShapes(approxC[0], approxC[0], CV_CONTOURS_MATCH_I3, 0.0);
	//cout << "hamming con se stessa " << hamming << endl;


	// READ IMAGE
	
	Mat image = imread("stella3.jpg");
	for (;;)
	{
		Mat result = star.findStarInImg(image, 92.0);
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

		Mat result = star.findStarInImg(image, 92.0);
		imshow("FINAL RESULT", result);

		if (waitKey(30) > 0)
			break;
	}
	*/
	
	
	waitKey(0);
	return 0;
}