#ifndef CALIB_COMMON_HPP
#define CALIB_COMMON_HPP
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <string>

namespace calib {
#define OVERLAY_DELAY 1000
#define IMAGE_MAX_WIDTH 1280
#define IMAGE_MAX_HEIGHT 720

	bool showOverlayMessage(const std::string& message);
	static const double sigmaMult = 1.96;
	static const std::string mainWindowName = "Calibration", gridWindowName = "Board locations";
	static const std::string consoleHelp = "Hot keys:\nesc - exit application\n"
		"s - save current data to .xml file \nr - delete last frame\n"
		"u - enable/disable applying undistortion \nd - delete all frames \nv - switch visualization\n\n";

	struct CapParams {
		//camera constants
		int camID = 700;
		cv::Size camRes = cv::Size(1280, 720);
		int fps = 20;
		int delay = 500;
		//board constants
		int charDictName = 10;
		cv::Size boardSz = cv::Size(5, 7);
		float sqrLen = 35.0f;
		float markLen = 16.3f;
		int markSz = 17;
		//calibration constants
		int calibStep = 1;
		int maxFramesNum = 30;
		int minFramesNum = 10;
		CapParams() = default;
	};

	//MATH to get other structs
	struct InternalParams {
		double solverEps = 1e-7;
		double alpha = .1;
		int solverMaxIters = 30;
		bool fastSolving = false;
		InternalParams() = default;
	};

	//CAMERA INTRINSICS
	struct CamParams {
		cv::Mat camMat, distCos, stdDevs;
		double avgErr = 0;
		CamParams() = default;
		CamParams(cv::Mat& _camMat, cv::Mat& _distCos, cv::Mat& _stdDevs, double _avgErr = 0) :
			camMat(_camMat), distCos(_distCos), stdDevs(_stdDevs), avgErr(_avgErr) {}
	};
	//FINAL DATA
	struct CalibData {
		cv::Mat camMat, distCos, stdDevs, perViewErrors;
		std::vector<cv::Mat> rvecs, tvecs;
		double totalAvgErr = 0;
		cv::Size imgSz = cv::Size(1280, 720);
		std::vector<std::vector<cv::Point2f>> imgPts;
		std::vector<std::vector<cv::Point3f>> objPts;
		std::vector<cv::Mat> allCharCorns, allCharIds;
		cv::Mat undistMap1, undistMap2;
		CalibData() = default;
	};
}
#endif