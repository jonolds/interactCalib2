#include "calibControl.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
using namespace std;
using namespace cv;

double calib::CalibControl::estimateCoverageQuality() {
	int gridSize = 10;
	int xGridStep = mCalibData->imageSize.width / gridSize;
	int yGridStep = mCalibData->imageSize.height / gridSize;
	vector<int> pointsInCell(gridSize * gridSize);
	fill(pointsInCell.begin(), pointsInCell.end(), 0);

	for (vector<vector<Point2f>>::iterator it = mCalibData->imagePoints.begin(); it != mCalibData->imagePoints.end(); ++it)
		for (vector<Point2f>::iterator pointIt = (*it).begin(); pointIt != (*it).end(); ++pointIt) {
			int i = (int)((*pointIt).x / xGridStep);
			int j = (int)((*pointIt).y / yGridStep);
			pointsInCell[i * gridSize + j]++;
		}
	for (vector<Mat>::iterator it = mCalibData->allCharucoCorners.begin(); it != mCalibData->allCharucoCorners.end(); ++it)
		for (int l = 0; l < (*it).size[0]; l++) {
			int i = (int)((*it).at<float>(l, 0) / xGridStep);
			int j = (int)((*it).at<float>(l, 1) / yGridStep);
			pointsInCell[i * gridSize + j]++;
		}
	Mat mean, stdDev;
	meanStdDev(pointsInCell, mean, stdDev);
	return mean.at<double>(0) / (stdDev.at<double>(0) + 1e-7);
}
calib::CalibControl::CalibControl() {
	mCalibFlags = 0;
}
calib::CalibControl::CalibControl(Ptr<CalibData> data, int initialFlags, bool autoTuning, int minFramesNum) :
	mCalibData(std::move(data)) {
	mCalibFlags = initialFlags;
	mNeedTuning = autoTuning;
	mMinFramesNum = minFramesNum;
	mConfIntervalsState = false;
	mCoverageQualityState = false;
}

void calib::CalibControl::updateState() {
	if (mCalibData->cameraMatrix.total()) {
		const double relErrEps = 0.05;
		bool fConfState = false, cConfState = false, dConfState = true;
		if (sigmaMult * mCalibData->stdDeviations.at<double>(0) / mCalibData->cameraMatrix.at<double>(0, 0) < relErrEps &&
			sigmaMult * mCalibData->stdDeviations.at<double>(1) / mCalibData->cameraMatrix.at<double>(1, 1) < relErrEps)
			fConfState = true;
		if (sigmaMult * mCalibData->stdDeviations.at<double>(2) / mCalibData->cameraMatrix.at<double>(0, 2) < relErrEps &&
			sigmaMult * mCalibData->stdDeviations.at<double>(3) / mCalibData->cameraMatrix.at<double>(1, 2) < relErrEps)
			cConfState = true;
		for (int i = 0; i < 5; i++)
			if (mCalibData->stdDeviations.at<double>(4 + i) / fabs(mCalibData->distCoeffs.at<double>(i)) > 1)
				dConfState = false;
		mConfIntervalsState = fConfState && cConfState && dConfState;
	}

	if (getFramesNumberState())
		mCoverageQualityState = estimateCoverageQuality() > 1.8;
	if (getFramesNumberState() && mNeedTuning) {
		if (!(mCalibFlags & CALIB_FIX_ASPECT_RATIO) &&
			mCalibData->cameraMatrix.total()) {
			double fDiff = fabs(mCalibData->cameraMatrix.at<double>(0, 0) -
				mCalibData->cameraMatrix.at<double>(1, 1));
			if (fDiff < 3 * mCalibData->stdDeviations.at<double>(0) &&
				fDiff < 3 * mCalibData->stdDeviations.at<double>(1)) {
				mCalibFlags |= CALIB_FIX_ASPECT_RATIO;
				mCalibData->cameraMatrix.at<double>(0, 0) =
					mCalibData->cameraMatrix.at<double>(1, 1);
			}
		}
		if (!(mCalibFlags & CALIB_ZERO_TANGENT_DIST)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCoeffs.at<double>(2)) < eps &&
				fabs(mCalibData->distCoeffs.at<double>(3)) < eps)
				mCalibFlags |= CALIB_ZERO_TANGENT_DIST;
		}
		if (!(mCalibFlags & CALIB_FIX_K1)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCoeffs.at<double>(0)) < eps)
				mCalibFlags |= CALIB_FIX_K1;
		}
		if (!(mCalibFlags & CALIB_FIX_K2)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCoeffs.at<double>(1)) < eps)
				mCalibFlags |= CALIB_FIX_K2;
		}
		if (!(mCalibFlags & CALIB_FIX_K3)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCoeffs.at<double>(4)) < eps)
				mCalibFlags |= CALIB_FIX_K3;
		}
	}
}

bool calib::CalibControl::getCommonCalibrationState() const {
	int rating = (int)getFramesNumberState() + (int)getConfidenceIntrervalsState() +
		(int)getRMSState() + (int)mCoverageQualityState;
	return rating == 4;
}
bool calib::CalibControl::getFramesNumberState() const {
	return max(mCalibData->imagePoints.size(), mCalibData->allCharucoCorners.size()) > mMinFramesNum;
}
bool calib::CalibControl::getConfidenceIntrervalsState() const {
	return mConfIntervalsState;
}
bool calib::CalibControl::getRMSState() const {
	return mCalibData->totalAvgErr < 0.5;
}
int calib::CalibControl::getNewFlags() const {
	return mCalibFlags;
}

//////////////////// CalibDataControl
double calib::CalibDataControl::estimateGridSubsetQuality(size_t excludedIndex) {
	{
		int gridSize = 10;
		int xGridStep = mCalibData->imageSize.width / gridSize;
		int yGridStep = mCalibData->imageSize.height / gridSize;
		vector<int> pointsInCell(gridSize * gridSize);
		fill(pointsInCell.begin(), pointsInCell.end(), 0);
		for (size_t k = 0; k < mCalibData->imagePoints.size(); k++)
			if (k != excludedIndex)
				for (vector<Point2f>::iterator pointIt = mCalibData->imagePoints[k].begin(); pointIt != mCalibData->imagePoints[k].
				     end(); ++pointIt) {
					int i = (int)((*pointIt).x / xGridStep);
					int j = (int)((*pointIt).y / yGridStep);
					pointsInCell[i * gridSize + j]++;
				}

		for (size_t k = 0; k < mCalibData->allCharucoCorners.size(); k++)
			if (k != excludedIndex)
				for (int l = 0; l < mCalibData->allCharucoCorners[k].size[0]; l++) {
					int i = (int)(mCalibData->allCharucoCorners[k].at<float>(l, 0) / xGridStep);
					int j = (int)(mCalibData->allCharucoCorners[k].at<float>(l, 1) / yGridStep);
					pointsInCell[i * gridSize + j]++;
				}
		Mat mean, stdDev;
		meanStdDev(pointsInCell, mean, stdDev);
		return mean.at<double>(0) / (stdDev.at<double>(0) + 1e-7);
	}
}

calib::CalibDataControl::CalibDataControl(Ptr<CalibData> data, int maxFrames, double convParameter) :
	mCalibData(move(data)), mParamsFileName("CamParams.xml") {
	mMaxFramesNum = maxFrames;
	mAlpha = convParameter;
}
calib::CalibDataControl::CalibDataControl() {}

void calib::CalibDataControl::filterFrames() {
	size_t numberOfFrames = max(mCalibData->allCharucoIds.size(), mCalibData->imagePoints.size());
	CV_Assert(numberOfFrames == mCalibData->perViewErrors.total());
	if (numberOfFrames >= mMaxFramesNum) {
		double worstValue = -HUGE_VAL, maxQuality = estimateGridSubsetQuality(numberOfFrames);
		size_t worstElemIndex = 0;
		for (size_t i = 0; i < numberOfFrames; i++) {
			double gridQDelta = estimateGridSubsetQuality(i) - maxQuality;
			double currentValue = mCalibData->perViewErrors.at<double>((int)i) * mAlpha + gridQDelta * (1. - mAlpha);
			if (currentValue > worstValue) {
				worstValue = currentValue;
				worstElemIndex = i;
			}
		}
		showOverlayMessage(format("Frame %d is worst", worstElemIndex + 1));
		if (!mCalibData->imagePoints.empty()) {
			mCalibData->imagePoints.erase(mCalibData->imagePoints.begin() + worstElemIndex);
			mCalibData->objectPoints.erase(mCalibData->objectPoints.begin() + worstElemIndex);
		}
		else {
			mCalibData->allCharucoCorners.erase(mCalibData->allCharucoCorners.begin() + worstElemIndex);
			mCalibData->allCharucoIds.erase(mCalibData->allCharucoIds.begin() + worstElemIndex);
		}

		Mat newErrorsVec = Mat((int)numberOfFrames - 1, 1, CV_64F);
		copy(mCalibData->perViewErrors.ptr<double>(0),
		          mCalibData->perViewErrors.ptr<double>((int)worstElemIndex), newErrorsVec.ptr<double>(0));
		copy(mCalibData->perViewErrors.ptr<double>((int)worstElemIndex + 1), mCalibData->perViewErrors.ptr<double>((int)numberOfFrames),
		          newErrorsVec.ptr<double>((int)worstElemIndex));
		mCalibData->perViewErrors = newErrorsVec;
	}
}

void calib::CalibDataControl::setParametersFileName(const std::string& name) {
	mParamsFileName = name;
}

void calib::CalibDataControl::deleteLastFrame() {
	if (!mCalibData->imagePoints.empty()) {
		mCalibData->imagePoints.pop_back();
		mCalibData->objectPoints.pop_back();
	}
	if (!mCalibData->allCharucoCorners.empty()) {
		mCalibData->allCharucoCorners.pop_back();
		mCalibData->allCharucoIds.pop_back();
	}
	if (!mParamsStack.empty()) {
		mCalibData->cameraMatrix = (mParamsStack.top()).cameraMatrix;
		mCalibData->distCoeffs = (mParamsStack.top()).distCoeffs;
		mCalibData->stdDeviations = (mParamsStack.top()).stdDeviations;
		mCalibData->totalAvgErr = (mParamsStack.top()).avgError;
		mParamsStack.pop();
	}
}

void calib::CalibDataControl::rememberCurrentParameters() {
	Mat oldCameraMat, oldDistcoeefs, oldStdDevs;
	mCalibData->cameraMatrix.copyTo(oldCameraMat);
	mCalibData->distCoeffs.copyTo(oldDistcoeefs);
	mCalibData->stdDeviations.copyTo(oldStdDevs);
	mParamsStack.push(cameraParameters(oldCameraMat, oldDistcoeefs, oldStdDevs, mCalibData->totalAvgErr));
}

void calib::CalibDataControl::deleteAllData() {
	mCalibData->imagePoints.clear();
	mCalibData->objectPoints.clear();
	mCalibData->allCharucoCorners.clear();
	mCalibData->allCharucoIds.clear();
	mCalibData->cameraMatrix = mCalibData->distCoeffs = Mat();
	mParamsStack = stack<cameraParameters>();
	rememberCurrentParameters();
}

bool calib::CalibDataControl::saveCurrentCameraParameters() const {
	bool success = false;
	if (mCalibData->cameraMatrix.total()) {
		FileStorage parametersWriter(mParamsFileName, FileStorage::WRITE);
		if (parametersWriter.isOpened()) {
			time_t rawtime;
			time(&rawtime);
			char buf[256];
			strftime(buf, sizeof(buf) - 1, "%c", localtime(&rawtime));

			parametersWriter << "calibrationDate" << buf;
			parametersWriter << "framesCount" << max((int)mCalibData->objectPoints.size(), (int)mCalibData->allCharucoCorners.size());
			parametersWriter << "cameraResolution" << mCalibData->imageSize;
			parametersWriter << "cameraMatrix" << mCalibData->cameraMatrix;
			parametersWriter << "cameraMatrix_std_dev" << mCalibData->stdDeviations.rowRange(Range(0, 4));
			parametersWriter << "dist_coeffs" << mCalibData->distCoeffs;
			parametersWriter << "dist_coeffs_std_dev" << mCalibData->stdDeviations.rowRange(Range(4, 9));
			parametersWriter << "avg_reprojection_error" << mCalibData->totalAvgErr;
			parametersWriter.release();
			success = true;
		}
	}
	return success;
}

void calib::CalibDataControl::printParametersToConsole(ostream& output) const {
	const char* border = "---------------------------------------------------";
	output << border << endl;
	output << "Frames used for calibration: " << max(mCalibData->objectPoints.size(), mCalibData->allCharucoCorners.size())
		<< " \t RMS = " << mCalibData->totalAvgErr << endl;
	if (mCalibData->cameraMatrix.at<double>(0, 0) == mCalibData->cameraMatrix.at<double>(1, 1))
		output << "F = " << mCalibData->cameraMatrix.at<double>(1, 1) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(1) << endl;
	else
		output << "Fx = " << mCalibData->cameraMatrix.at<double>(0, 0) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(0) << " \t "
			<< "Fy = " << mCalibData->cameraMatrix.at<double>(1, 1) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(1) << endl;
	output << "Cx = " << mCalibData->cameraMatrix.at<double>(0, 2) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(2) << " \t"
		<< "Cy = " << mCalibData->cameraMatrix.at<double>(1, 2) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(3) << endl;
	output << "K1 = " << mCalibData->distCoeffs.at<double>(0) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(4) << endl;
	output << "K2 = " << mCalibData->distCoeffs.at<double>(1) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(5) << endl;
	output << "K3 = " << mCalibData->distCoeffs.at<double>(4) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(8) << endl;
	output << "TD1 = " << mCalibData->distCoeffs.at<double>(2) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(6) << endl;
	output << "TD2 = " << mCalibData->distCoeffs.at<double>(3) << " +- " << sigmaMult * mCalibData->stdDeviations.at<double>(7) << endl;
}

void calib::CalibDataControl::updateUndistortMap() {
	initUndistortRectifyMap(mCalibData->cameraMatrix, mCalibData->distCoeffs, noArray(),
		getOptimalNewCameraMatrix(mCalibData->cameraMatrix, mCalibData->distCoeffs, mCalibData->imageSize, 0.0, 
		mCalibData->imageSize), mCalibData->imageSize, CV_16SC2, mCalibData->undistMap1, mCalibData->undistMap2);
}