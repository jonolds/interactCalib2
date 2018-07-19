#include "FrameProc.hpp"
#include "rotationConverters.hpp"
#include "calibCommon.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>
//#include <limits>
using namespace calib;
using namespace std;
using namespace cv;

#define VIDEO_TEXT_SIZE 4
#define POINT_SIZE 5
bool CalibProc::detectParseChAruco(const Mat& frame) {
	Ptr<aruco::Board> board = mCharBoard.staticCast<aruco::Board>();
	vector<vector<Point2f>> corners, rejected;
	vector<int> ids;
	detectMarkers(frame, mArucoDict, corners, ids, aruco::DetectorParameters::create(), rejected);
	//refineDetectedMarkers(frame, board, corners, ids, rejected);
	Mat curCharCorns, curCharIds;
	if (!ids.empty())
		interpolateCornersCharuco(corners, ids, frame, mCharBoard, curCharCorns, curCharIds);
	if (!ids.empty()) aruco::drawDetectedMarkers(frame, corners);
	if (curCharCorns.total() > 3) {
		float centerX = 0, centerY = 0;
		for (int i = 0; i < curCharCorns.size[0]; i++) {
			centerX += curCharCorns.at<float>(i, 0);
			centerY += curCharCorns.at<float>(i, 1);
		}
		centerX /= curCharCorns.size[0];
		centerY /= curCharCorns.size[0];
		mTemplateLoc.insert(mTemplateLoc.begin(), Point2f(centerX, centerY));
		aruco::drawDetectedCornersCharuco(frame, curCharCorns, curCharIds);
		mCurCharCorns = curCharCorns;
		mCurCharIds = curCharIds;
		return true;
	}
	return false;
}

void CalibProc::saveFrameData() {
	vector<Point3f> objectPoints;
	mCalibData->allCharCorns.push_back(mCurCharCorns);
	mCalibData->allCharIds.push_back(mCurCharIds);
}

void CalibProc::showCaptMsg(const Mat& frame, const string& message) {
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
	if (!mCalibData->camMat.total()) {
		tmpCamMatrix = Mat::eye(3, 3, CV_64F);
		tmpCamMatrix.at<double>(0, 0) = 20000;
		tmpCamMatrix.at<double>(1, 1) = 20000;
		tmpCamMatrix.at<double>(0, 2) = (double)mCalibData->imgSz.height / 2;
		tmpCamMatrix.at<double>(1, 2) = (double)mCalibData->imgSz.width / 2;
	}
	else {
		Mat r, t, angles;
		vector<Point3f> allObjPoints;
		allObjPoints.reserve(mCurCharIds.total());
		for (int i = 0; i < (int)mCurCharIds.total(); i++) {
			int pointID = mCurCharIds.at<int>((int)i);
			CV_Assert(pointID >= 0 && pointID < (int)mCharBoard->chessboardCorners.size());
			allObjPoints.push_back(mCharBoard->chessboardCorners[pointID]);
		}
		solvePnP(allObjPoints, mCurCharCorns, tmpCamMatrix, mCalibData->distCos, r, t);
		RodriguesToEuler(r, angles, CALIB_DEGREES);
		if (180.0 - fabs(angles.at<double>(0)) > badAngleThresh || fabs(angles.at<double>(1)) > badAngleThresh) {
			isFrameBad = true;
			mCalibData->allCharCorns.pop_back();
			mCalibData->allCharIds.pop_back();
		}
	}
	return isFrameBad;
}

CalibProc::CalibProc(Ptr<CalibData> data, CapParams& capParams) : mCalibData(std::move(data)), mBoardSize(capParams.boardSz) {
	mCaptFrames = 0;
	mNeededFramesNum = capParams.calibStep;
	mCapDelay = static_cast<int>(capParams.capDelay * capParams.fps);
	mMaxTemplateOffset = sqrt(static_cast<float>(mCalibData->imgSz.height * mCalibData->imgSz.height) +
		static_cast<float>(mCalibData->imgSz.width * mCalibData->imgSz.width)) / 20.0;
	mSqrSz = capParams.markLen;
	mArucoDict = getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(capParams.charDictName));
	mCharBoard = aruco::CharucoBoard::create(mBoardSize.width, mBoardSize.height, capParams.sqrLen,
		(float)capParams.charMarkSz, mArucoDict);
}

Mat CalibProc::processFrame(const Mat& frame) {
	Mat frameCopy = frame.clone();
	mCurImgPts.clear();
	bool isTemplateFound = detectParseChAruco(frameCopy);
	if (mTemplateLoc.size() > mCapDelay)
		mTemplateLoc.pop_back();
	if (mTemplateLoc.size() == mCapDelay && isTemplateFound) {
		if (norm(mTemplateLoc.front() - mTemplateLoc.back()) < mMaxTemplateOffset) {
			saveFrameData();
			bool isFrameBad = checkLastFrame();
			if (!isFrameBad) {
				string dispMsg = format("Frame # %d captured\n", max(mCalibData->imgPts.size(), mCalibData->allCharCorns.size()));
				if (!showOverlayMessage(dispMsg))
					showCaptMsg(frame, dispMsg);
				mCaptFrames++;
			}
			else {
				if (!showOverlayMessage("Frame rejected\n"))
					showCaptMsg(frame, "Frame rejected\n");
			}
			mTemplateLoc.clear();
			mTemplateLoc.reserve(mCapDelay);
		}
	}
	return frameCopy;
}

bool CalibProc::isProcessed() const { return mCaptFrames >= mNeededFramesNum; }
void CalibProc::resetState() { mCaptFrames = 0; mTemplateLoc.clear(); }
///////////////////////////////////////////
void ShowProc::drawBoard(Mat& img, InputArray points) {
	Mat tmpView = Mat::zeros(img.rows, img.cols, CV_8UC3);
	vector<Point2f> templateHull;
	vector<Point> poly;
	convexHull(points, templateHull);
	poly.resize(templateHull.size());
	for (int i = 0; i < (int)templateHull.size(); i++)
		poly[i] = Point((int)(templateHull[i].x * mGridVwScale), (int)(templateHull[i].y * mGridVwScale));
	fillConvexPoly(tmpView, poly, Scalar(0, 255, 0), LINE_AA);
	addWeighted(tmpView, .2, img, 1, 0, img);
}
void ShowProc::drawGridPoints(const Mat& frame) {
	for(vector<Mat>::iterator it = mCaldata->allCharCorns.begin(); it!=mCaldata->allCharCorns.end(); ++it)
		for (int i = 0; i < (*it).size[0]; i++)
			circle(frame, Point((int)(*it).at<float>(i, 0), (int)(*it).at<float>(i, 1)),
			       POINT_SIZE, Scalar(0, 255, 0), 1, LINE_AA);
}

ShowProc::ShowProc(Ptr<CalibData> data, Ptr<CalibControl> controller) :
	mCaldata(move(data)), mController(move(controller)) {
	mNeedUndistort = true;
	mVisMode = Grid;
	mGridVwScale = 0.5;
	mTextSize = VIDEO_TEXT_SIZE;
}

Mat ShowProc::processFrame(const Mat& frame) {
	if (!mCaldata->camMat.empty() && !mCaldata->distCos.empty()) {
		mTextSize = VIDEO_TEXT_SIZE * (double)frame.cols / IMAGE_MAX_WIDTH;
		Scalar textColor = Scalar(0, 0, 255);
		Mat frameCopy;
		if (mNeedUndistort && mController->getFramesNumberState()) {
			if (mVisMode == Grid)
				drawGridPoints(frame);
			remap(frame, frameCopy, mCaldata->undistMap1, mCaldata->undistMap2, INTER_LINEAR);
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
		if (mCaldata->stdDevs.at<double>(0) == 0)
			displayMessage = format("F = %d RMS = %.3f", (int)mCaldata->camMat.at<double>(0, 0), mCaldata->totalAvgErr);
		else
			displayMessage = format("Fx = %d Fy = %d RMS = %.3f", (int)mCaldata->camMat.at<double>(0, 0),
			                        (int)mCaldata->camMat.at<double>(1, 1), mCaldata->totalAvgErr);
		if (mController->getRMSState() && mController->getFramesNumberState())
			displayMessage.append(" OK");
		int baseLine = 100;
		Size textSize = getTextSize(displayMessage, 1, mTextSize - 1, 2, &baseLine);
		Point textOrigin = Point(baseLine, 2 * textSize.height);
		putText(frameCopy, displayMessage, textOrigin, 1, mTextSize - 1, textColor, 2, LINE_AA);

		if (mCaldata->stdDevs.at<double>(0) == 0)
			displayMessage = format("DF = %.2f", mCaldata->stdDevs.at<double>(1) * sigmaMult);
		else
			displayMessage = format("DFx = %.2f DFy = %.2f", mCaldata->stdDevs.at<double>(0) * sigmaMult,
			                        mCaldata->stdDevs.at<double>(1) * sigmaMult);
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
			displayMessage.append(format("AR=%.3f ", mCaldata->camMat.at<double>(0, 0) / mCaldata->camMat.at<double>(1, 1)));
		if (calibFlags & CALIB_ZERO_TANGENT_DIST)
			displayMessage.append("TD=0 ");
		displayMessage.append(format("K1=%.2f K2=%.2f K3=%.2f", mCaldata->distCos.at<double>(0), mCaldata->distCos.at<double>(1),
		                             mCaldata->distCos.at<double>(4)));
		putText(frameCopy, displayMessage, Point(baseLine, frameCopy.rows - (int)(1.5 * textSize.height)),
		        1, mTextSize - 1, textColor, 2, LINE_AA);
		return frameCopy;
	}
	return frame;
}
bool ShowProc::isProcessed() const { return false; }
void ShowProc::setVisualizationMode(visualisationMode mode) { mVisMode = mode; }
void ShowProc::switchVisualizationMode() {
	if (mVisMode == Grid) {
		mVisMode = Window;
		namedWindow(gridWindowName); moveWindow(gridWindowName, 1280, 500);
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
		Size originSize = mCaldata->imgSz;
		Mat altGridView = Mat::zeros((int)(originSize.height * mGridVwScale), (int)(originSize.width * mGridVwScale), CV_8UC3);
		for (vector<Mat>::iterator it = mCaldata->allCharCorns.begin(); it != mCaldata->allCharCorns.end(); ++it)
			drawBoard(altGridView, *it);
		imshow(gridWindowName, altGridView);
	}
}
void ShowProc::switchUndistort() { mNeedUndistort = !mNeedUndistort; }
void ShowProc::setUndistort(bool isEnabled) { mNeedUndistort = isEnabled; }