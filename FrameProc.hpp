#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP
#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include "calibCommon.hpp"
#include "calibControl.hpp"

namespace calib {
	class FrameProc {
	public:
		virtual ~FrameProc();
		virtual cv::Mat processFrame(const cv::Mat& frame) = 0;
		virtual bool isProcessed() const = 0;
		virtual void resetState() = 0;
	};

	class CalibProc : public FrameProc {
	protected:
		cv::Ptr<CalibData> mCalibData;
		TemplateType mBoardType;
		cv::Size mBoardSize;
		std::vector<cv::Point2f> mTemplateLocations, mCurrentImagePoints;
		cv::Mat mCurrentCharucoCorners, mCurrentCharucoIds;
		cv::Ptr<cv::SimpleBlobDetector> mBlobDetectorPtr;

#ifdef HAVE_OPENCV_ARUCO
		cv::Ptr<cv::aruco::Dictionary> mArucoDictionary;
		cv::Ptr<cv::aruco::CharucoBoard> mCharucoBoard;
#endif
		int mNeededFramesNum, mCapuredFrames;
		unsigned mDelayBetweenCaptures;
		double mMaxTemplateOffset;
		float mSquareSize, mTemplDist;

		bool detectAndParseChessboard(const cv::Mat& frame);
		bool detectAndParseChAruco(const cv::Mat& frame);

		void saveFrameData();
		void showCaptureMessage(const cv::Mat& frame, const std::string& message);
		bool checkLastFrame();

	public:
		CalibProc(cv::Ptr<CalibData> data, CaptureParams& capParams);
		cv::Mat processFrame(const cv::Mat& frame) override;
		bool isProcessed() const override;
		void resetState() override;
		~CalibProc() override;
	};

	enum visualisationMode { Grid, Window };

	class ShowProc : public FrameProc {
	protected:
		cv::Ptr<CalibData> mCalibdata;
		cv::Ptr<CalibControl> mController;
		TemplateType mBoardType;
		visualisationMode mVisMode;
		bool mNeedUndistort;
		double mGridViewScale;
		double mTextSize;

		void drawBoard(cv::Mat& img, cv::InputArray points);
		void drawGridPoints(const cv::Mat& frame);
	public:
		ShowProc(cv::Ptr<CalibData> data, cv::Ptr<CalibControl> controller, TemplateType board);
		cv::Mat processFrame(const cv::Mat& frame) override;
		bool isProcessed() const override;
		void resetState() override;
		void setVisualizationMode(visualisationMode mode);
		void switchVisualizationMode();
		void clearBoardsView();
		void updateBoardsView();
		void switchUndistort();
		void setUndistort(bool isEnabled);
		~ShowProc() override;
	};
}
#endif