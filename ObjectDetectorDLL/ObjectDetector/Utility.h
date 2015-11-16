#ifdef OBJECTDETECTOR_EXPORTS
#define OBJECTDETECTOR_API __declspec(dllexport) 
#else
#define OBJECTDETECTOR_API __declspec(dllimport) 
#endif

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/shape.hpp"

namespace od
{
	class Utility{

		

	public:

		enum CentroidDetectionMode{
			ONE_LOOP,
			TWO_LOOP,
			THREE_LOOP
		};

		enum SplitMode{
			MODE_1,
			MODE_2
		};


		static OBJECTDETECTOR_API double calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base);


		static OBJECTDETECTOR_API double correlationWithBase(std::vector<cv::Point> contourKeypoints, std::vector<cv::Point> baseKeypoints);


		
		static OBJECTDETECTOR_API void findCentroidsKeypoints(std::vector<cv::Point> contour,
														std::vector<cv::Point>& centroids,
														CentroidDetectionMode mode);

	private:

		OBJECTDETECTOR_API static std::vector<cv::Rect> splitRect(cv::Rect box, cv::Point centroid, int level, SplitMode mode);

		OBJECTDETECTOR_API static double spearmanCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base);

		OBJECTDETECTOR_API static std::vector<double> findDistancesFromCenter(std::vector<cv::Point> distribution);

		OBJECTDETECTOR_API static std::vector<double> findAnglesRespectCenter(std::vector<cv::Point> distribution);

	};
}