#include "calibControl.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>


double calib::CalibControl::estimateCoverageQuality() {
	
	int gridSize = 10;
	int xGridStep = mCalibData->imgSz.width / gridSize;
	int yGridStep = mCalibData->imgSz.height / gridSize;
	std::vector<int> pointsInCell(100, 0);

	for (std::vector<std::vector<cv::Point2f>>::iterator it = mCalibData->imgPts.begin(); it != mCalibData->imgPts.end(); ++it)
		for (std::vector<cv::Point2f>::iterator pointIt = (*it).begin(); pointIt != (*it).end(); ++pointIt) {
			int i = (int)((*pointIt).x / xGridStep);
			int j = (int)((*pointIt).y / yGridStep);
			pointsInCell[i * gridSize + j]++;
		}
	for (std::vector<cv::Mat>::iterator it = mCalibData->allCharCorns.begin(); it != mCalibData->allCharCorns.end(); ++it)
		for (int l = 0; l < (*it).size[0]; l++) {
			int i = (int)((*it).at<float>(l, 0) / xGridStep);
			int j = (int)((*it).at<float>(l, 1) / yGridStep);
			pointsInCell[i * gridSize + j]++;
		}
	cv::Mat mean, stdDev;
	meanStdDev(pointsInCell, mean, stdDev);
	return mean.at<double>(0) / (stdDev.at<double>(0) + 1e-7);
}
calib::CalibControl::CalibControl() {
	mCalibFlags = 0;
}
calib::CalibControl::CalibControl(cv::Ptr<CalibData> data, int initialFlags, bool autoTuning, int minFramesNum) :
	mCalibData(std::move(data)) {
	mCalibFlags = initialFlags;
	mNeedTuning = autoTuning;
	mMinFramesNum = minFramesNum;
	mConfIntervalsState = false;
	mCoverageQualityState = false;
}

void calib::CalibControl::updateState() {
	if (mCalibData->camMat.total()) {
		const double relErrEps = 0.05;
		bool fConfState = false, cConfState = false, dConfState = true;
		if (sigmaMult * mCalibData->stdDevs.at<double>(0) / mCalibData->camMat.at<double>(0, 0) < relErrEps &&
			sigmaMult * mCalibData->stdDevs.at<double>(1) / mCalibData->camMat.at<double>(1, 1) < relErrEps)
			fConfState = true;
		if (sigmaMult * mCalibData->stdDevs.at<double>(2) / mCalibData->camMat.at<double>(0, 2) < relErrEps &&
			sigmaMult * mCalibData->stdDevs.at<double>(3) / mCalibData->camMat.at<double>(1, 2) < relErrEps)
			cConfState = true;
		for (int i = 0; i < 5; i++)
			if (mCalibData->stdDevs.at<double>(4 + i) / fabs(mCalibData->distCos.at<double>(i)) > 1)
				dConfState = false;
		mConfIntervalsState = fConfState && cConfState && dConfState;
	}

	if (getFramesNumberState())
		mCoverageQualityState = estimateCoverageQuality() > 1.8;
	if (getFramesNumberState() && mNeedTuning) {
		if (!(mCalibFlags & cv::CALIB_FIX_ASPECT_RATIO) &&
			mCalibData->camMat.total()) {
			double fDiff = fabs(mCalibData->camMat.at<double>(0, 0) -
				mCalibData->camMat.at<double>(1, 1));
			if (fDiff < 3 * mCalibData->stdDevs.at<double>(0) && fDiff < 3 * mCalibData->stdDevs.at<double>(1)) {
				mCalibFlags |= cv::CALIB_FIX_ASPECT_RATIO;
				mCalibData->camMat.at<double>(0, 0) = mCalibData->camMat.at<double>(1, 1);
			}
		}
		if (!(mCalibFlags & cv::CALIB_ZERO_TANGENT_DIST)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCos.at<double>(2)) < eps && fabs(mCalibData->distCos.at<double>(3)) < eps)
				mCalibFlags |= cv::CALIB_ZERO_TANGENT_DIST;
		}
		if (!(mCalibFlags & cv::CALIB_FIX_K1)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCos.at<double>(0)) < eps)
				mCalibFlags |= cv::CALIB_FIX_K1;
		}
		if (!(mCalibFlags & cv::CALIB_FIX_K2)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCos.at<double>(1)) < eps)
				mCalibFlags |= cv::CALIB_FIX_K2;
		}
		if (!(mCalibFlags & cv::CALIB_FIX_K3)) {
			const double eps = 0.005;
			if (fabs(mCalibData->distCos.at<double>(4)) < eps)
				mCalibFlags |= cv::CALIB_FIX_K3;
		}
	}
}

bool calib::CalibControl::getCommonCalibrationState() const {
	int rating = (int)getFramesNumberState() + (int)getConfidenceIntrervalsState() +
		(int)getRMSState() + (int)mCoverageQualityState;
	return rating == 4;
}
bool calib::CalibControl::getFramesNumberState() const {
	return std::max(mCalibData->imgPts.size(), mCalibData->allCharCorns.size()) > mMinFramesNum;
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
double calib::CalibDataControl::estimateGridSubsetQuality(unsigned excludedIndex) {
	{
		int gridSize = 10;
		int xGridStep = mCalibData->imgSz.width / gridSize;
		int yGridStep = mCalibData->imgSz.height / gridSize;
		std::vector<int> pointsInCell(gridSize * gridSize, 0);
		//fill(pointsInCell.begin(), pointsInCell.end(), 0);
		for (unsigned k = 0; k < mCalibData->imgPts.size(); k++)
			if (k != excludedIndex)
				for (std::vector<cv::Point2f>::iterator pointIt = mCalibData->imgPts[k].begin(); pointIt != mCalibData->imgPts[k].
				     end(); ++pointIt) {
					int i = (int)((*pointIt).x / xGridStep);
					int j = (int)((*pointIt).y / yGridStep);
					pointsInCell[i * gridSize + j]++;
				}

		for (unsigned k = 0; k < mCalibData->allCharCorns.size(); k++)
			if (k != excludedIndex)
				for (int l = 0; l < mCalibData->allCharCorns[k].size[0]; l++) {
					int i = (int)(mCalibData->allCharCorns[k].at<float>(l, 0) / xGridStep);
					int j = (int)(mCalibData->allCharCorns[k].at<float>(l, 1) / yGridStep);
					pointsInCell[i * gridSize + j]++;
				}
		cv::Mat mean, stdDev;
		meanStdDev(pointsInCell, mean, stdDev);
		return mean.at<double>(0) / (stdDev.at<double>(0) + 1e-7);
	}
}

calib::CalibDataControl::CalibDataControl(cv::Ptr<CalibData> data, int maxFrames, double convParameter) :
	mCalibData(std::move(data)), mParamsFileName("CamParams.xml") {
	mMaxFramesNum = maxFrames;
	mAlpha = convParameter;
}
calib::CalibDataControl::CalibDataControl() = default;

void calib::CalibDataControl::filterFrames() {
	unsigned numberOfFrames = (unsigned)std::max(mCalibData->allCharIds.size(), mCalibData->imgPts.size());
	CV_Assert(numberOfFrames == mCalibData->perViewErrors.total());
	if (numberOfFrames >= mMaxFramesNum) {
		double worstValue = -HUGE_VAL, maxQuality = estimateGridSubsetQuality(numberOfFrames);
		unsigned worstElemIndex = 0;
		for (unsigned i = 0; i < numberOfFrames; i++) {
			double gridQDelta = estimateGridSubsetQuality(i) - maxQuality;
			double currentValue = mCalibData->perViewErrors.at<double>((int)i) * mAlpha + gridQDelta * (1. - mAlpha);
			if (currentValue > worstValue) {
				worstValue = currentValue;
				worstElemIndex = i;
			}
		}
		showOverlayMessage(cv::format("Frame %d is worst\n", worstElemIndex + 1));
		if (!mCalibData->imgPts.empty()) {
			mCalibData->imgPts.erase(mCalibData->imgPts.begin() + worstElemIndex);
			mCalibData->objPts.erase(mCalibData->objPts.begin() + worstElemIndex);
		}
		else {
			mCalibData->allCharCorns.erase(mCalibData->allCharCorns.begin() + worstElemIndex);
			mCalibData->allCharIds.erase(mCalibData->allCharIds.begin() + worstElemIndex);
		}

		cv::Mat newErrorsVec = cv::Mat((int)numberOfFrames - 1, 1, CV_64F);
		std::copy(mCalibData->perViewErrors.ptr<double>(0), mCalibData->perViewErrors.ptr<double>((int)worstElemIndex), newErrorsVec.ptr<double>(0));
		std::copy(mCalibData->perViewErrors.ptr<double>((int)worstElemIndex + 1), mCalibData->perViewErrors.ptr<double>((int)numberOfFrames),
		          newErrorsVec.ptr<double>((int)worstElemIndex));
		mCalibData->perViewErrors = newErrorsVec;
	}
}

void calib::CalibDataControl::setParametersFileName(const std::string& name) {
	mParamsFileName = name;
}

void calib::CalibDataControl::deleteLastFrame() {
	if (!mCalibData->imgPts.empty()) {
		mCalibData->imgPts.pop_back();
		mCalibData->objPts.pop_back();
	}
	if (!mCalibData->allCharCorns.empty()) {
		mCalibData->allCharCorns.pop_back();
		mCalibData->allCharIds.pop_back();
	}
	if (!mParamsStack.empty()) {
		mCalibData->camMat = (mParamsStack.top()).camMat;
		mCalibData->distCos = (mParamsStack.top()).distCos;
		mCalibData->stdDevs = (mParamsStack.top()).stdDevs;
		mCalibData->totalAvgErr = (mParamsStack.top()).avgErr;
		mParamsStack.pop();
	}
}

void calib::CalibDataControl::rememberCurrentParameters() {
	cv::Mat oldCameraMat, oldDistcoeefs, oldStdDevs;
	mCalibData->camMat.copyTo(oldCameraMat);
	mCalibData->distCos.copyTo(oldDistcoeefs);
	mCalibData->stdDevs.copyTo(oldStdDevs);
	mParamsStack.push(CamParams(oldCameraMat, oldDistcoeefs, oldStdDevs, mCalibData->totalAvgErr));
}

void calib::CalibDataControl::deleteAllData() {
	mCalibData->imgPts.clear();
	mCalibData->objPts.clear();
	mCalibData->allCharCorns.clear();
	mCalibData->allCharIds.clear();
	mCalibData->camMat = mCalibData->distCos = cv::Mat();
	mParamsStack = std::stack<CamParams>();
	rememberCurrentParameters();
}

bool calib::CalibDataControl::saveCurrentCameraParameters() const {
	bool success = false;
	if (mCalibData->camMat.total()) {
		cv::FileStorage parametersWriter(mParamsFileName, cv::FileStorage::WRITE);
		if (parametersWriter.isOpened()) {
			time_t rawtime;
			time(&rawtime);
			char buf[256];
			strftime(buf, sizeof(buf) - 1, "%c", localtime(&rawtime));

			parametersWriter << "calibrationDate" << buf;
			parametersWriter << "framesCount" << std::max((int)mCalibData->objPts.size(), (int)mCalibData->allCharCorns.size());
			parametersWriter << "cameraResolution" << mCalibData->imgSz;
			parametersWriter << "camMat" << mCalibData->camMat;
			parametersWriter << "cameraMatrix_std_dev" << mCalibData->stdDevs.rowRange(cv::Range(0, 4));
			parametersWriter << "dist_coeffs" << mCalibData->distCos;
			parametersWriter << "dist_coeffs_std_dev" << mCalibData->stdDevs.rowRange(cv::Range(4, 9));
			parametersWriter << "avg_reprojection_error" << mCalibData->totalAvgErr;
			parametersWriter.release();
			success = true;
		}
	}
	return success;
}

void calib::CalibDataControl::printParametersToConsole(std::ostream& output) const {
	const char* border = "---------------------------------------------------";
	output << border << std::endl;
	output << "Frames used for calibration: " << std::max(mCalibData->objPts.size(), mCalibData->allCharCorns.size())
		<< " \t RMS = " << mCalibData->totalAvgErr << std::endl;
	if (mCalibData->camMat.at<double>(0, 0) == mCalibData->camMat.at<double>(1, 1))
		output << "F = " << mCalibData->camMat.at<double>(1, 1) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(1) << std::endl;
	else
		output << "Fx = " << mCalibData->camMat.at<double>(0, 0) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(0) << " \t "
			<< "Fy = " << mCalibData->camMat.at<double>(1, 1) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(1) << std::endl;
	output << "Cx = " << mCalibData->camMat.at<double>(0, 2) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(2) << " \t"
		<< "Cy = " << mCalibData->camMat.at<double>(1, 2) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(3) << std::endl;
	output << "K1 = " << mCalibData->distCos.at<double>(0) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(4) << std::endl;
	output << "K2 = " << mCalibData->distCos.at<double>(1) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(5) << std::endl;
	output << "K3 = " << mCalibData->distCos.at<double>(4) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(8) << std::endl;
	output << "TD1 = " << mCalibData->distCos.at<double>(2) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(6) << std::endl;
	output << "TD2 = " << mCalibData->distCos.at<double>(3) << " +- " << sigmaMult * mCalibData->stdDevs.at<double>(7) << std::endl;
}

void calib::CalibDataControl::updateUndistortMap() {
	cv::Mat camMat = getOptimalNewCameraMatrix(mCalibData->camMat, mCalibData->distCos, mCalibData->imgSz, 0.0, mCalibData->imgSz);
	initUndistortRectifyMap(mCalibData->camMat, mCalibData->distCos, cv::noArray(),
		camMat, mCalibData->imgSz, CV_16SC2, mCalibData->undistMap1, mCalibData->undistMap2);
}