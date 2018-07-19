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
		virtual Mat processFrame(const Mat& frame) = 0;
		virtual bool isProcessed() const = 0;
		virtual void resetState() = 0;
	};

	class CalibProc : public FrameProc {
	protected:
		Ptr<CalibData> mCalibData;
		Size mBoardSize;
		std::vector<Point2f> mTemplateLoc, mCurImgPts;
		Mat mCurCharCorns, mCurCharIds;
		Ptr<SimpleBlobDetector> mBlobDetectPtr;

		Ptr<aruco::Dictionary> mArucoDict;
		Ptr<aruco::CharucoBoard> mCharBoard;
		int mNeededFramesNum, mCaptFrames;
		int delay;
		double mMaxTemplateOffset;
		float mSqrSz, mTemplDist;
		bool detectParseChAruco(const Mat& frame);
		void saveFrameData();
		void showCaptMsg(const Mat& frame, const std::string& message);
		bool checkLastFrame();
	public:
		CalibProc(Ptr<CalibData> data);
		Mat processFrame(const Mat& frame) override;
		bool isProcessed() const override;
		void resetState() override;
		~CalibProc() override = default;
	};

	enum visualisationMode { Grid, Window };

	class ShowProc : public FrameProc {
	protected:
		Ptr<CalibData> mCaldata;
		Ptr<CalibControl> mController;
		visualisationMode mVisMode;
		bool mNeedUndistort;
		double mGridVwScale;
		double mTextSize;
		void drawBoard(Mat& img, InputArray points);
		void drawGridPoints(const Mat& frame);
	public:
		ShowProc(Ptr<CalibData> data, Ptr<CalibControl> controller);
		Mat processFrame(const Mat& frame) override;
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