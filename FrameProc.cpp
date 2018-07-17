#include "FrameProc.hpp"
#include "rotationConverters.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
using namespace calib;
using namespace std;
using namespace cv;

#define VIDEO_TEXT_SIZE 4
#define POINT_SIZE 5
FrameProc::~FrameProc() {}
bool CalibProc::detectAndParseChessboard(const Mat& frame) {
	int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;
	bool isTemplateFound = findChessboardCorners(frame, mBoardSize, mCurrentImagePoints, chessBoardFlags);
	if (isTemplateFound) {
		Mat viewGray;
		cvtColor(frame, viewGray, COLOR_BGR2GRAY);
		cornerSubPix(viewGray, mCurrentImagePoints, Size(11, 11),
		             Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
		drawChessboardCorners(frame, mBoardSize, Mat(mCurrentImagePoints), isTemplateFound);
		mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
	}
	return isTemplateFound;
}

bool CalibProc::detectAndParseChAruco(const Mat& frame) {
#ifdef HAVE_OPENCV_ARUCO
	Ptr<aruco::Board> board = mCharucoBoard.staticCast<aruco::Board>();
	vector<vector<Point2f>> corners, rejected;
	vector<int> ids;
	detectMarkers(frame, mArucoDictionary, corners, ids, aruco::DetectorParameters::create(), rejected);
	//refineDetectedMarkers(frame, board, corners, ids, rejected);
	Mat currentCharucoCorners, currentCharucoIds;
	if (!ids.empty())
		interpolateCornersCharuco(corners, ids, frame, mCharucoBoard, currentCharucoCorners, currentCharucoIds);
	if (!ids.empty()) aruco::drawDetectedMarkers(frame, corners);
	if (currentCharucoCorners.total() > 3) {
		float centerX = 0, centerY = 0;
		for (int i = 0; i < currentCharucoCorners.size[0]; i++) {
			centerX += currentCharucoCorners.at<float>(i, 0);
			centerY += currentCharucoCorners.at<float>(i, 1);
		}
		centerX /= currentCharucoCorners.size[0];
		centerY /= currentCharucoCorners.size[0];
		mTemplateLocations.insert(mTemplateLocations.begin(), Point2f(centerX, centerY));
		aruco::drawDetectedCornersCharuco(frame, currentCharucoCorners, currentCharucoIds);
		mCurrentCharucoCorners = currentCharucoCorners;
		mCurrentCharucoIds = currentCharucoIds;
		return true;
	}
#else
    (void)frame;
#endif
	return false;
}

void CalibProc::saveFrameData() {
	vector<Point3f> objectPoints;
	switch (mBoardType) {
	case Chessboard:
		objectPoints.reserve(mBoardSize.height * mBoardSize.width);
		for (int i = 0; i < mBoardSize.height; ++i)
			for (int j = 0; j < mBoardSize.width; ++j)
				objectPoints.emplace_back(j * mSquareSize, i * mSquareSize, (float)0);
		mCalibData->imagePoints.push_back(mCurrentImagePoints);
		mCalibData->objectPoints.push_back(objectPoints);
		break;
	case chAruco:
		mCalibData->allCharucoCorners.push_back(mCurrentCharucoCorners);
		mCalibData->allCharucoIds.push_back(mCurrentCharucoIds);
		break;
	}
}

void CalibProc::showCaptureMessage(const Mat& frame, const string& message) {
	Point textOrigin(100, 100);
	double textSize = VIDEO_TEXT_SIZE * frame.cols / (double)IMAGE_MAX_WIDTH;
	bitwise_not(frame, frame);
	putText(frame, message, textOrigin, 1, textSize, Scalar(0, 0, 255), 2, LINE_AA);
	imshow(mainWindowName, frame);
	waitKey(300);
}

bool CalibProc::checkLastFrame() {
	bool isFrameBad = false;
	Mat tmpCamMatrix;
	const double badAngleThresh = 40;
	if (!mCalibData->cameraMatrix.total()) {
		tmpCamMatrix = Mat::eye(3, 3, CV_64F);
		tmpCamMatrix.at<double>(0, 0) = 20000;
		tmpCamMatrix.at<double>(1, 1) = 20000;
		tmpCamMatrix.at<double>(0, 2) = mCalibData->imageSize.height / 2;
		tmpCamMatrix.at<double>(1, 2) = mCalibData->imageSize.width / 2;
	}
	else
		mCalibData->cameraMatrix.copyTo(tmpCamMatrix);
	if (mBoardType != chAruco) {
		Mat r, t, angles;
		solvePnP(mCalibData->objectPoints.back(), mCurrentImagePoints, tmpCamMatrix, mCalibData->distCoeffs, r, t);
		RodriguesToEuler(r, angles, CALIB_DEGREES);
		if (fabs(angles.at<double>(0)) > badAngleThresh || fabs(angles.at<double>(1)) > badAngleThresh) {
			mCalibData->objectPoints.pop_back();
			mCalibData->imagePoints.pop_back();
			isFrameBad = true;
		}
	}
	else {
#ifdef HAVE_OPENCV_ARUCO
		Mat r, t, angles;
		vector<Point3f> allObjPoints;
		allObjPoints.reserve(mCurrentCharucoIds.total());
		for (size_t i = 0; i < mCurrentCharucoIds.total(); i++) {
			int pointID = mCurrentCharucoIds.at<int>((int)i);
			CV_Assert(pointID >= 0 && pointID < (int)mCharucoBoard->chessboardCorners.size());
			allObjPoints.push_back(mCharucoBoard->chessboardCorners[pointID]);
		}
		solvePnP(allObjPoints, mCurrentCharucoCorners, tmpCamMatrix, mCalibData->distCoeffs, r, t);
		RodriguesToEuler(r, angles, CALIB_DEGREES);
		if (180.0 - fabs(angles.at<double>(0)) > badAngleThresh || fabs(angles.at<double>(1)) > badAngleThresh) {
			isFrameBad = true;
			mCalibData->allCharucoCorners.pop_back();
			mCalibData->allCharucoIds.pop_back();
		}
#endif
	}
	return isFrameBad;
}

CalibProc::CalibProc(Ptr<CalibData> data, CaptureParams& capParams) :
	mCalibData(std::move(data)), mBoardType(capParams.board), mBoardSize(capParams.boardSize) {
	mCapuredFrames = 0;
	mNeededFramesNum = capParams.calibrationStep;
	mDelayBetweenCaptures = static_cast<int>(capParams.captureDelay * capParams.fps);
	mMaxTemplateOffset = sqrt(static_cast<float>(mCalibData->imageSize.height * mCalibData->imageSize.height) +
		static_cast<float>(mCalibData->imageSize.width * mCalibData->imageSize.width)) / 20.0;
	mSquareSize = capParams.squareSize;
	mTemplDist = (float)capParams.templDst;
	switch (mBoardType) {
	case chAruco:
#ifdef HAVE_OPENCV_ARUCO
		mArucoDictionary = getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(capParams.charucoDictName));
		mCharucoBoard = aruco::CharucoBoard::create(mBoardSize.width, mBoardSize.height, capParams.charucoSquareLength,
		                                            (float)capParams.charucoMarkerSize, mArucoDictionary);
#endif
		break;
	case Chessboard:
		break;
	}
}

Mat CalibProc::processFrame(const Mat& frame) {
	Mat frameCopy;
	frame.copyTo(frameCopy);
	bool isTemplateFound = false;
	mCurrentImagePoints.clear();
	switch (mBoardType) {
	case Chessboard:
		isTemplateFound = detectAndParseChessboard(frameCopy);
		break;
	case chAruco:
		isTemplateFound = detectAndParseChAruco(frameCopy);
		break;
	}
	if (mTemplateLocations.size() > mDelayBetweenCaptures)
		mTemplateLocations.pop_back();
	if (mTemplateLocations.size() == mDelayBetweenCaptures && isTemplateFound) {
		if (norm(mTemplateLocations.front() - mTemplateLocations.back()) < mMaxTemplateOffset) {
			saveFrameData();
			bool isFrameBad = checkLastFrame();
			if (!isFrameBad) {
				string displayMessage = format("Frame # %d captured", std::max(mCalibData->imagePoints.size(),
				                                                                    mCalibData->allCharucoCorners.size()));
				if (!showOverlayMessage(displayMessage))
					showCaptureMessage(frame, displayMessage);
				mCapuredFrames++;
			}
			else {
				string displayMessage = "Frame rejected";
				if (!showOverlayMessage(displayMessage))
					showCaptureMessage(frame, displayMessage);
			}
			mTemplateLocations.clear();
			mTemplateLocations.reserve(mDelayBetweenCaptures);
		}
	}
	return frameCopy;
}

bool CalibProc::isProcessed() const { return mCapuredFrames >= mNeededFramesNum; }
void CalibProc::resetState() { mCapuredFrames = 0; mTemplateLocations.clear(); }
CalibProc::~CalibProc() {}
///////////////////////////////////////////
void ShowProc::drawBoard(Mat& img, InputArray points) {
	Mat tmpView = Mat::zeros(img.rows, img.cols, CV_8UC3);
	vector<Point2f> templateHull;
	vector<Point> poly;
	convexHull(points, templateHull);
	poly.resize(templateHull.size());
	for (size_t i = 0; i < templateHull.size(); i++)
		poly[i] = Point((int)(templateHull[i].x * mGridViewScale), (int)(templateHull[i].y * mGridViewScale));
	fillConvexPoly(tmpView, poly, Scalar(0, 255, 0), LINE_AA);
	addWeighted(tmpView, .2, img, 1, 0, img);
}
void ShowProc::drawGridPoints(const Mat& frame) {
	if (mBoardType != chAruco)
		for (vector<vector<Point2f>>::iterator it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it)
			for (vector<Point2f>::iterator pointIt = (*it).begin(); pointIt != (*it).end(); ++pointIt)
				circle(frame, *pointIt, POINT_SIZE, Scalar(0, 255, 0), 1, LINE_AA);
	else
		for (vector<Mat>::iterator it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
			for (int i = 0; i < (*it).size[0]; i++)
				circle(frame, Point((int)(*it).at<float>(i, 0), (int)(*it).at<float>(i, 1)),
				       POINT_SIZE, Scalar(0, 255, 0), 1, LINE_AA);
}

ShowProc::ShowProc(Ptr<CalibData> data, Ptr<CalibControl> controller, TemplateType board) :
	mCalibdata(move(data)), mController(move(controller)), mBoardType(board) {
	mNeedUndistort = true;
	mVisMode = Grid;
	mGridViewScale = 0.5;
	mTextSize = VIDEO_TEXT_SIZE;
}

Mat ShowProc::processFrame(const Mat& frame) {
	if (!mCalibdata->cameraMatrix.empty() && !mCalibdata->distCoeffs.empty()) {
		mTextSize = VIDEO_TEXT_SIZE * (double)frame.cols / IMAGE_MAX_WIDTH;
		Scalar textColor = Scalar(0, 0, 255);
		Mat frameCopy;
		if (mNeedUndistort && mController->getFramesNumberState()) {
			if (mVisMode == Grid)
				drawGridPoints(frame);
			remap(frame, frameCopy, mCalibdata->undistMap1, mCalibdata->undistMap2, INTER_LINEAR);
			int baseLine = 100;
			Size textSize = getTextSize("Undistorted view", 1, mTextSize, 2, &baseLine);
			Point textOrigin(baseLine, frame.rows - (int)(2.5 * textSize.height));
			putText(frameCopy, "Undistorted view", textOrigin, 1, mTextSize, textColor, 2, LINE_AA);
		}
		else {
			frame.copyTo(frameCopy);
			if (mVisMode == Grid)
				drawGridPoints(frameCopy);
		}
		string displayMessage;
		if (mCalibdata->stdDeviations.at<double>(0) == 0)
			displayMessage = format("F = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0, 0), mCalibdata->totalAvgErr);
		else
			displayMessage = format("Fx = %d Fy = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0, 0),
			                        (int)mCalibdata->cameraMatrix.at<double>(1, 1), mCalibdata->totalAvgErr);
		if (mController->getRMSState() && mController->getFramesNumberState())
			displayMessage.append(" OK");
		int baseLine = 100;
		Size textSize = getTextSize(displayMessage, 1, mTextSize - 1, 2, &baseLine);
		Point textOrigin = Point(baseLine, 2 * textSize.height);
		putText(frameCopy, displayMessage, textOrigin, 1, mTextSize - 1, textColor, 2, LINE_AA);

		if (mCalibdata->stdDeviations.at<double>(0) == 0)
			displayMessage = format("DF = %.2f", mCalibdata->stdDeviations.at<double>(1) * sigmaMult);
		else
			displayMessage = format("DFx = %.2f DFy = %.2f", mCalibdata->stdDeviations.at<double>(0) * sigmaMult,
			                        mCalibdata->stdDeviations.at<double>(1) * sigmaMult);
		if (mController->getConfidenceIntrervalsState() && mController->getFramesNumberState())
			displayMessage.append(" OK");
		putText(frameCopy, displayMessage, Point(baseLine, 4 * textSize.height), 1, mTextSize - 1, textColor, 2, LINE_AA);

		if (mController->getCommonCalibrationState()) {
			displayMessage = format("Calibration is done");
			putText(frameCopy, displayMessage, Point(baseLine, 6 * textSize.height), 1, mTextSize - 1, textColor, 2, LINE_AA);
		}
		int calibFlags = mController->getNewFlags();
		displayMessage = "";
		if (!(calibFlags & CALIB_FIX_ASPECT_RATIO))
			displayMessage.append(format("AR=%.3f ", mCalibdata->cameraMatrix.at<double>(0, 0) / mCalibdata->cameraMatrix.at<double>(1, 1)));
		if (calibFlags & CALIB_ZERO_TANGENT_DIST)
			displayMessage.append("TD=0 ");
		displayMessage.append(format("K1=%.2f K2=%.2f K3=%.2f", mCalibdata->distCoeffs.at<double>(0), mCalibdata->distCoeffs.at<double>(1),
		                             mCalibdata->distCoeffs.at<double>(4)));
		putText(frameCopy, displayMessage, Point(baseLine, frameCopy.rows - (int)(1.5 * textSize.height)),
		        1, mTextSize - 1, textColor, 2, LINE_AA);
		return frameCopy;
	}
	return frame;
}
bool ShowProc::isProcessed() const { return false; }
void ShowProc::resetState() {}
void ShowProc::setVisualizationMode(visualisationMode mode) { mVisMode = mode; }
void ShowProc::switchVisualizationMode() {
	if (mVisMode == Grid) {
		mVisMode = Window;
		updateBoardsView();
	}
	else {
		mVisMode = Grid;
		destroyWindow(gridWindowName);
	}
}
void ShowProc::clearBoardsView() { imshow(gridWindowName, Mat()); }
void ShowProc::updateBoardsView() {
	if (mVisMode == Window) {
		Size originSize = mCalibdata->imageSize;
		Mat altGridView = Mat::zeros((int)(originSize.height * mGridViewScale), (int)(originSize.width * mGridViewScale), CV_8UC3);
		if (mBoardType != chAruco)
			for (vector<vector<Point2f>>::iterator it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it) {
				size_t pointsNum = (*it).size() / 2;
				vector<Point2f> points(pointsNum);
				copy((*it).begin(), (*it).begin() + pointsNum, points.begin());
				drawBoard(altGridView, points);
				copy((*it).begin() + pointsNum, (*it).begin() + 2 * pointsNum, points.begin());
				drawBoard(altGridView, points);
			}
		else
			for (vector<Mat>::iterator it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
				drawBoard(altGridView, *it);
		imshow(gridWindowName, altGridView);
	}
}
void ShowProc::switchUndistort() { mNeedUndistort = !mNeedUndistort; }
void ShowProc::setUndistort(bool isEnabled) { mNeedUndistort = isEnabled; }
ShowProc::~ShowProc() {}