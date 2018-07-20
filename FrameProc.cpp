#include "FrameProc.hpp"
#include "rotationConverters.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include <vector>
#include <string>
#include <algorithm>
#include <limits>
using namespace calib;

#define VIDEO_TEXT_SIZE 4
#define POINT_SIZE 5
bool CalibProc::detectParseChAruco(const cv::Mat& frame) {
	cv::Ptr<cv::aruco::Board> board = mCharBoard.staticCast<cv::aruco::Board>();
	std::vector<std::vector<cv::Point2f>> corners, rejected;
	std::vector<int> ids;
	detectMarkers(frame, mArucoDict, corners, ids, cv::aruco::DetectorParameters::create(), rejected);
	//refineDetectedMarkers(frame, board, corners, ids, rejected);
	cv::Mat curCharCorns, curCharIds;
	if (!ids.empty())
		interpolateCornersCharuco(corners, ids, frame, mCharBoard, curCharCorns, curCharIds);
	if (!ids.empty()) cv::aruco::drawDetectedMarkers(frame, corners);
	if (curCharCorns.total() > 3) {
		float centerX = 0, centerY = 0;
		for (int i = 0; i < curCharCorns.size[0]; i++) {
			centerX += curCharCorns.at<float>(i, 0);
			centerY += curCharCorns.at<float>(i, 1);
		}
		centerX /= curCharCorns.size[0];
		centerY /= curCharCorns.size[0];
		mTemplateLoc.insert(mTemplateLoc.begin(), cv::Point2f(centerX, centerY));
		cv::aruco::drawDetectedCornersCharuco(frame, curCharCorns, curCharIds);
		mCurCharCorns = curCharCorns;
		mCurCharIds = curCharIds;
		return true;
	}
	return false;
}

void CalibProc::saveFrameData() {
	std::vector<cv::Point3f> objectPoints;
	mCalibData->allCharCorns.push_back(mCurCharCorns);
	mCalibData->allCharIds.push_back(mCurCharIds);
}

void CalibProc::showCaptMsg(const cv::Mat& frame, const std::string& message) {
	cv::Point textOrigin(100, 100);
	double textSize = VIDEO_TEXT_SIZE * frame.cols / (double)IMAGE_MAX_WIDTH;
	cv::bitwise_not(frame, frame);
	cv::putText(frame, message, textOrigin, 1, textSize, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	cv::imshow(mainWindowName, frame);
	cv::waitKey(300);
}

bool CalibProc::checkLastFrame() {
	bool isFrameBad = false;
	cv::Mat tmpCamMatrix;
	const double badAngleThresh = 40;
	if (!mCalibData->camMat.total()) {
		tmpCamMatrix = cv::Mat::eye(3, 3, CV_64F);
		tmpCamMatrix.at<double>(0, 0) = 20000;
		tmpCamMatrix.at<double>(1, 1) = 20000;
		tmpCamMatrix.at<double>(0, 2) = (double)mCalibData->imgSz.height / 2;
		tmpCamMatrix.at<double>(1, 2) = (double)mCalibData->imgSz.width / 2;
	}
	else {
		cv::Mat r, t, angles;
		std::vector<cv::Point3f> allObjPoints;
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

CalibProc::CalibProc(cv::Ptr<CalibData> data, CapParams& capParams) : mCalibData(std::move(data)), mBoardSize(capParams.boardSz) {
	mCaptFrames = 0;
	mNeededFramesNum = capParams.calibStep;
	mDelay = static_cast<int>(capParams.delay * capParams.fps);
	mMaxTemplateOffset = sqrt(static_cast<float>(mCalibData->imgSz.height * mCalibData->imgSz.height) +
		static_cast<float>(mCalibData->imgSz.width * mCalibData->imgSz.width)) / 20.0;
	mSqrSz = capParams.markLen;
	mArucoDict = getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(capParams.charDictName));
	mCharBoard = cv::aruco::CharucoBoard::create(mBoardSize.width, mBoardSize.height, capParams.sqrLen,
		(float)capParams.markSz, mArucoDict);
}

cv::Mat CalibProc::processFrame(const cv::Mat& frame) {
	cv::Mat frameCopy = frame.clone();
	mCurImgPts.clear();
	bool isTemplateFound = detectParseChAruco(frameCopy);
	if (mTemplateLoc.size() > mDelay)
		mTemplateLoc.pop_back();
	if (mTemplateLoc.size() == mDelay && isTemplateFound) {
		if (norm(mTemplateLoc.front() - mTemplateLoc.back()) < mMaxTemplateOffset) {
			saveFrameData();
			bool isFrameBad = checkLastFrame();
			if (!isFrameBad) {
				std::string dispMsg = cv::format("Frame # %d captured\n", std::max(mCalibData->imgPts.size(), mCalibData->allCharCorns.size()));
				if (!showOverlayMessage(dispMsg))
					showCaptMsg(frame, dispMsg);
				mCaptFrames++;
			}
			else {
				if (!showOverlayMessage("Frame rejected\n"))
					showCaptMsg(frame, "Frame rejected\n");
			}
			mTemplateLoc.clear();
			mTemplateLoc.reserve(mDelay);
		}
	}
	return frameCopy;
}

bool CalibProc::isProcessed() const { return mCaptFrames >= mNeededFramesNum; }
void CalibProc::resetState() { mCaptFrames = 0; mTemplateLoc.clear(); }
///////////////////////////////////////////
void ShowProc::drawBoard(cv::Mat& img, cv::InputArray points) {
	cv::Mat tmpView = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
	std::vector<cv::Point2f> templateHull;
	std::vector<cv::Point> poly;
	cv::convexHull(points, templateHull);
	poly.resize(templateHull.size());
	for (int i = 0; i < (int)templateHull.size(); i++)
		poly[i] = cv::Point((int)(templateHull[i].x * mGridVwScale), (int)(templateHull[i].y * mGridVwScale));
	fillConvexPoly(tmpView, poly, cv::Scalar(0, 255, 0), cv::LINE_AA);
	cv::addWeighted(tmpView, .2, img, 1, 0, img);
}
void ShowProc::drawGridPoints(const cv::Mat& frame) {
	for(std::vector<cv::Mat>::iterator it = mCaldata->allCharCorns.begin(); it!=mCaldata->allCharCorns.end(); ++it)
		for (int i = 0; i < (*it).size[0]; i++)
			circle(frame, cv::Point((int)(*it).at<float>(i, 0), (int)(*it).at<float>(i, 1)),
			       POINT_SIZE, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
}

ShowProc::ShowProc(cv::Ptr<CalibData> data, cv::Ptr<CalibControl> controller) :
	mCaldata(std::move(data)), mController(std::move(controller)) {
	mNeedUndistort = true;
	mVisMode = Grid;
	mGridVwScale = 0.5;
	mTextSize = VIDEO_TEXT_SIZE;
}

cv::Mat ShowProc::processFrame(const cv::Mat& frame) {
	if (!mCaldata->camMat.empty() && !mCaldata->distCos.empty()) {
		mTextSize = VIDEO_TEXT_SIZE * (double)frame.cols / IMAGE_MAX_WIDTH;
		cv::Scalar textColor = cv::Scalar(0, 0, 255);
		cv::Mat frameCopy;
		if (mNeedUndistort && mController->getFramesNumberState()) {
			if (mVisMode == Grid)
				drawGridPoints(frame);
			remap(frame, frameCopy, mCaldata->undistMap1, mCaldata->undistMap2, cv::INTER_LINEAR);
			int baseLine = 100;
			cv::Size textSize = cv::getTextSize("Undistorted view", 1, mTextSize, 2, &baseLine);
			cv::Point textOrigin(baseLine, frame.rows - (int)(2.5 * textSize.height));
			putText(frameCopy, "Undistorted view", textOrigin, 1, mTextSize, textColor, 2, cv::LINE_AA);
		}
		else {
			frame.copyTo(frameCopy);
			if (mVisMode == Grid)
				drawGridPoints(frameCopy);
		}
		std::string displayMessage;
		if (mCaldata->stdDevs.at<double>(0) == 0)
			displayMessage = cv::format("F = %d RMS = %.3f", (int)mCaldata->camMat.at<double>(0, 0), mCaldata->totalAvgErr);
		else
			displayMessage = cv::format("Fx = %d Fy = %d RMS = %.3f", (int)mCaldata->camMat.at<double>(0, 0),
			                        (int)mCaldata->camMat.at<double>(1, 1), mCaldata->totalAvgErr);
		if (mController->getRMSState() && mController->getFramesNumberState())
			displayMessage.append(" OK");
		int baseLine = 100;
		cv::Size textSize = cv::getTextSize(displayMessage, 1, mTextSize - 1, 2, &baseLine);
		cv::Point textOrigin = cv::Point(baseLine, 2 * textSize.height);
		putText(frameCopy, displayMessage, textOrigin, 1, mTextSize - 1, textColor, 2, cv::LINE_AA);

		if (mCaldata->stdDevs.at<double>(0) == 0)
			displayMessage = cv::format("DF = %.2f", mCaldata->stdDevs.at<double>(1) * sigmaMult);
		else
			displayMessage = cv::format("DFx = %.2f DFy = %.2f", mCaldata->stdDevs.at<double>(0) * sigmaMult,
			                        mCaldata->stdDevs.at<double>(1) * sigmaMult);
		if (mController->getConfidenceIntrervalsState() && mController->getFramesNumberState())
			displayMessage.append(" OK");
		cv::putText(frameCopy, displayMessage, cv::Point(baseLine, 4 * textSize.height), 1, mTextSize - 1, textColor, 2, cv::LINE_AA);

		if (mController->getCommonCalibrationState()) {
			displayMessage = cv::format("Calibration is done");
			cv::putText(frameCopy, displayMessage, cv::Point(baseLine, 6 * textSize.height), 1, mTextSize - 1, textColor, 2, cv::LINE_AA);
		}
		int calibFlags = mController->getNewFlags();
		displayMessage = "";
		if (!(calibFlags & cv::CALIB_FIX_ASPECT_RATIO))
			displayMessage.append(cv::format("AR=%.3f ", mCaldata->camMat.at<double>(0, 0) / mCaldata->camMat.at<double>(1, 1)));
		if (calibFlags & cv::CALIB_ZERO_TANGENT_DIST)
			displayMessage.append("TD=0 ");
		displayMessage.append(cv::format("K1=%.2f K2=%.2f K3=%.2f", mCaldata->distCos.at<double>(0), mCaldata->distCos.at<double>(1),
		                             mCaldata->distCos.at<double>(4)));
		cv::putText(frameCopy, displayMessage, cv::Point(baseLine, frameCopy.rows - (int)(1.5 * textSize.height)),
		        1, mTextSize - 1, textColor, 2, cv::LINE_AA);
		return frameCopy;
	}
	return frame;
}
bool ShowProc::isProcessed() const { return false; }
void ShowProc::setVisualizationMode(visualisationMode mode) { mVisMode = mode; }
void ShowProc::switchVisualizationMode() {
	if (mVisMode == Grid) {
		mVisMode = Window;
		cv::namedWindow(gridWindowName);
		cv::moveWindow(gridWindowName, 1280, 500);
		updateBoardsView();
	}
	else {
		mVisMode = Grid;
		cv::destroyWindow(gridWindowName);
	}
}
void ShowProc::clearBoardsView() { cv::imshow(gridWindowName, cv::Mat()); }
void ShowProc::updateBoardsView() {
	if (mVisMode == Window) {
		cv::Size originSize = mCaldata->imgSz;
		cv::Mat altGridView = cv::Mat::zeros((int)(originSize.height * mGridVwScale), (int)(originSize.width * mGridVwScale), CV_8UC3);
		for (std::vector<cv::Mat>::iterator it = mCaldata->allCharCorns.begin(); it != mCaldata->allCharCorns.end(); ++it)
			drawBoard(altGridView, *it);
		imshow(gridWindowName, altGridView);
	}
}
void ShowProc::switchUndistort() { mNeedUndistort = !mNeedUndistort; }
void ShowProc::setUndistort(bool isEnabled) { mNeedUndistort = isEnabled; }