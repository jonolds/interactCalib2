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
	enum InputType { Video, Pictures };
	enum InputVideoSource { Camera, File };
	enum TemplateType { Chessboard, chAruco };
	static const std::string mainWindowName = "Calibration", gridWindowName = "Board locations";
	static const std::string consoleHelp = "Hot keys:\nesc - exit application\n"
		"s - save current data to .xml file \nr - delete last frame\n"
		"u - enable/disable applying undistortion \nd - delete all frames \nv - switch visualization";

	static const double sigmaMult = 1.96;
	struct CalibData {
		cv::Mat cameraMatrix, distCoeffs, stdDeviations, perViewErrors;
		std::vector<cv::Mat> rvecs, tvecs;
		double totalAvgErr;
		cv::Size imageSize;

		std::vector<std::vector<cv::Point2f>> imagePoints;
		std::vector<std::vector<cv::Point3f>> objectPoints;
		std::vector<cv::Mat> allCharucoCorners, allCharucoIds;
		cv::Mat undistMap1, undistMap2;

		CalibData() {
			imageSize = cv::Size(IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT);
		}
	};

	struct cameraParameters {
		cv::Mat cameraMatrix, distCoeffs, stdDeviations;
		double avgError;
		cameraParameters() {}
		cameraParameters(cv::Mat& _cameraMatrix, cv::Mat& _distCoeffs, cv::Mat& _stdDeviations, double _avgError = 0) :
			cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), stdDeviations(_stdDeviations), avgError(_avgError) {}
	};

	struct CaptureParams {
		InputType captureMethod;
		InputVideoSource source;
		TemplateType board;
		cv::Size boardSize;
		int charucoDictName;
		int calibrationStep;
		float charucoSquareLength;
		int charucoMarkerSize;
		int captureDelay; 
		float squareSize;
		int templDst;
		std::string videoFileName;
		int camID, fps;
		cv::Size cameraResolution;
		int maxFramesNum, minFramesNum;

		CaptureParams() {
			captureMethod = Video;
			source = Camera;
			board = chAruco;
			boardSize = cv::Size(5, 3);   //?????????????????????????
			charucoDictName = 10;   //?????????????????????????
			charucoSquareLength = 35;  //?????????????????????????
			charucoMarkerSize = 17;
			squareSize = (float)16.3;
			templDst = 295; //***Circle method
			videoFileName = "";
			camID = 700;
			calibrationStep = 1;
			captureDelay = 500;
			maxFramesNum = 30;
			minFramesNum = 10;
			fps = 20;
			cameraResolution = cv::Size(1280, 720);
		}
	};

	struct InternalParams {
		double solverEps, filterAlpha;
		int solverMaxIters;
		bool fastSolving;

		InternalParams() {
			solverEps = 1e-7;
			solverMaxIters = 30;
			fastSolving = false;
			filterAlpha = 0.1;
		}
	};
}
#endif