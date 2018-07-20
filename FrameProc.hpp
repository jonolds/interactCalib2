#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include "calibCommon.hpp"
#include "calibControl.hpp"

namespace calib {
	class FrameProc {
	public:
		virtual ~FrameProc() = default;
		virtual cv::Mat processFrame(const cv::Mat& frame) = 0;
		virtual bool isProcessed() const = 0;
		virtual void resetState() = 0;
	};

	class CalibProc : public FrameProc {
	protected:
		cv::Ptr<CalibData> mCalibData;
		cv::Size mBoardSize;
		std::vector<cv::Point2f> mTemplateLoc, mCurImgPts;
		cv::Mat mCurCharCorns, mCurCharIds;
		cv::Ptr<cv::SimpleBlobDetector> mBlobDetectPtr;

		cv::Ptr<cv::aruco::Dictionary> mArucoDict;
		cv::Ptr<cv::aruco::CharucoBoard> mCharBoard;
		int mNeededFramesNum, mCaptFrames;
		unsigned mDelay;
		double mMaxTemplateOffset;
		float mSqrSz, mTemplDist;
		bool detectParseChAruco(const cv::Mat& frame);
		void saveFrameData();
		void showCaptMsg(const cv::Mat& frame, const std::string& message);
		bool checkLastFrame();
	public:
		CalibProc(cv::Ptr<CalibData> data, CapParams& capParams);
		cv::Mat processFrame(const cv::Mat& frame) override;
		bool isProcessed() const override;
		void resetState() override;
		~CalibProc() override = default;
	};

	enum visualisationMode { Grid, Window };

	class ShowProc : public FrameProc {
	protected:
		cv::Ptr<CalibData> mCaldata;
		cv::Ptr<CalibControl> mController;
		visualisationMode mVisMode;
		bool mNeedUndistort;
		double mGridVwScale;
		double mTextSize;
		void drawBoard(cv::Mat& img, cv::InputArray points);
		void drawGridPoints(const cv::Mat& frame);
	public:
		ShowProc(cv::Ptr<CalibData> data, cv::Ptr<CalibControl> controller);
		cv::Mat processFrame(const cv::Mat& frame) override;
		bool isProcessed() const override;
		void resetState() override {}
		void setVisualizationMode(visualisationMode mode);
		void switchVisualizationMode();
		void clearBoardsView();
		void updateBoardsView();
		void switchUndistort();
		void setUndistort(bool isEnabled);
		~ShowProc() override = default;
	};
}
#endif