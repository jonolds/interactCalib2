#ifndef CALIB_CONTROLLER_HPP
#define CALIB_CONTROLLER_HPP
#include "calibCommon.hpp"
#include <stack>
#include <string>
#include <ostream>

namespace calib {
	class CalibControl {
	protected:
		cv::Ptr<CalibData> mCalibData;
		int mCalibFlags;
		unsigned mMinFramesNum;
		bool mNeedTuning;
		bool mConfIntervalsState;
		bool mCoverageQualityState;

		double estimateCoverageQuality();
	public:
		CalibControl();
		CalibControl(cv::Ptr<CalibData> data, int initialFlags, bool autoTuning, int minFramesNum);

		void updateState();
		bool getCommonCalibrationState() const;
		bool getFramesNumberState() const;
		bool getConfidenceIntrervalsState() const;
		bool getRMSState() const;
		int getNewFlags() const;
	};

	class CalibDataControl {
	protected:
		cv::Ptr<CalibData> mCalibData;
		std::stack<CamParams> mParamsStack;
		std::string mParamsFileName;
		unsigned mMaxFramesNum;
		double mAlpha;
		double estimateGridSubsetQuality(unsigned excludedIndex);
	public:
		CalibDataControl(cv::Ptr<CalibData> data, int maxFrames, double convParameter);
		CalibDataControl();

		void filterFrames();
		void setParametersFileName(const std::string& name);
		void deleteLastFrame();
		void rememberCurrentParameters();
		void deleteAllData();
		bool saveCurrentCameraParameters() const;
		void printParametersToConsole(std::ostream& output) const;
		void updateUndistortMap();
	};
}
#endif