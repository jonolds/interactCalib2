#ifndef CALIB_COMMON_HPP
#define CALIB_COMMON_HPP
#include <opencv2/core.hpp>
//#include <memory>
#include <vector>
#include <string>
#include <opencv2/aruco/charuco.hpp>

using namespace std;
using namespace cv;

namespace calib {
#define OVERLAY_DELAY 1000
#define IMAGE_MAX_WIDTH 1280
#define IMAGE_MAX_HEIGHT 720

	struct CapParams {
		//camera constants
		int camID = 700;
		Size camRes = Size(1280, 720);
		int fps = 20;
		int capDelay = 500;
		//board constants
		int charDictName = 10;
		Size boardSz = Size(5, 7);
		float sqrLen = 35.0f;
		float markLen = 16.3f;
		int charMarkSz = 17;
		//calibration constants
		int calibStep = 1;
		int maxFramesNum = 30;
		int minFramesNum = 10;
		CapParams() = default;
	};

	bool showOverlayMessage(const string& message);
	Ptr<aruco::CharucoBoard> makePrintBoard();
	static const double sigmaMult = 1.96;
	static const string mainWindowName = "Calibration", gridWindowName = "Board locations";
	static const string consoleHelp = "Hot keys:\nesc - exit application\n"
		"s - save current data to .xml file \nr - delete last frame\n"
		"u - enable/disable applying undistortion \nd - delete all frames \nv - switch visualization\n\n";

	//FINAL DATA
	struct CalibData {
		Mat camMat, distCos, stdDevs, perViewErrors;
		vector<Mat> rvecs, tvecs;
		double totalAvgErr = 0;
		Size imgSz = Size(1280, 720);
		vector<vector<Point2f>> imgPts;
		vector<vector<Point3f>> objPts;
		vector<Mat> allCharCorns, allCharIds;
		Mat undistMap1, undistMap2;
		CalibData() = default;
	};

	//CAMERA INTRINSICS
	struct CamParams {
		Mat camMat, distCos, stdDevs;
		double avgError = 0;
		CamParams() = default;

		CamParams(Mat& _cameraMatrix, Mat& _distCoeffs, Mat& _stdDeviations, double _avgError = 0) :
			camMat(_cameraMatrix), distCos(_distCoeffs), stdDevs(_stdDeviations), avgError(_avgError) {}
	};

	//MATH to get other structs
	struct InternalParams {
		double solverEps = 1e-7;
		double alpha = .1;
		int solverMaxIters = 30;
		bool fastSolving = false;
		InternalParams() = default;
	};
}
#endif