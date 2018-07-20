#ifndef CALIB_PIPELINE_HPP
#define CALIB_PIPELINE_HPP
#include <vector>
#include <opencv2/highgui.hpp>
#include "calibCommon.hpp"
#include "FrameProc.hpp"

namespace calib {
	enum PipelineExitStatus {
		Finished, DeleteLastFrame, Calibrate, DeleteAllFrames,
		SaveCurrentData, SwitchUndistort, SwitchVisualisation
	};
	class CalibPipeline {
	protected:
		CapParams capParams;
		cv::Size mImageSize;
		cv::VideoCapture cap;
		cv::Size getCameraResolution();
	public:
		explicit CalibPipeline(CapParams params);
		PipelineExitStatus start(std::vector<cv::Ptr<FrameProc>> processors);
		cv::Size getImageSize() const;
	};
}
#endif