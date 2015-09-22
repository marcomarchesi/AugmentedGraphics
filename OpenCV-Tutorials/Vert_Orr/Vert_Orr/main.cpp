#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char)
{
	Mat src = imread("foto.jpg");

	namedWindow("src", WINDOW_AUTOSIZE);
	//namedWindow("src2", WINDOW_AUTOSIZE);
	imshow("src", src);

	Mat gray;
	if (src.channels() == 3)
	{
		cvtColor(src, gray, CV_BGR2GRAY);
	}
	else
		gray = src;

	imshow("src", gray);

	Mat bw;
	adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	Mat bw2;
	adaptiveThreshold(gray, bw2, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);

	//imshow("src2", bw2);
	imshow("src", bw);

	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();

	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / 30;

	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));

	imshow("horizontal", horizontal);
	//---------------------------------------------------------------------------------------

	int verticalsize = vertical.rows / 30;

	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));

	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	// Show extracted vertical lines

	imshow("vertical", vertical);


	// Extract edges and smooth image according to the logic
	// 1. extract edges
	// 2. dilate(edges)
	// 3. src.copyTo(smooth)
	// 4. blur smooth img
	// 5. smooth.copyTo(src, edges)

	// Step 1
	Mat edges;
	adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
	imshow("edges", edges);

	// Step 2
	Mat kernel = Mat::ones(2, 2, CV_8UC1);
	dilate(edges, edges, kernel);
	imshow("dilate", edges);

	// Step 3
	Mat smooth;
	vertical.copyTo(smooth);

	// Step 4
	blur(smooth, smooth, Size(2, 2));

	// Step 5
	smooth.copyTo(vertical, edges);

	// Show final result
	imshow("smooth", vertical);

	waitKey(0);
	return 0;
}