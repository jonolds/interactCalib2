#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace calib;
using namespace std;
using namespace cv;

#define CAP_DELAY 10
Size CalibPipeline::getCameraResolution() {
	cap.set(CAP_PROP_FRAME_WIDTH, 10000);
	cap.set(CAP_PROP_FRAME_HEIGHT, 10000);
	int w = (int)cap.get(CAP_PROP_FRAME_WIDTH), h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	return Size(w, h);
}

CalibPipeline::CalibPipeline(CaptureParams params) : mCaptureParams(std::move(params)) {}

PipelineExitStatus CalibPipeline::start(vector<Ptr<FrameProc>> processors) {
	if (mCaptureParams.source == Camera && !cap.isOpened()) {
		cap.open(mCaptureParams.camID);
		Size maxRes = getCameraResolution();
		Size neededRes = mCaptureParams.cameraResolution;

		if (maxRes.width < neededRes.width) {
			double aR = (double)maxRes.width / maxRes.height;
			cap.set(CAP_PROP_FRAME_WIDTH, neededRes.width);
			cap.set(CAP_PROP_FRAME_HEIGHT, neededRes.width / aR);
		}
		else if (maxRes.height < neededRes.height) {
			double aR = (double)maxRes.width / maxRes.height;
			cap.set(CAP_PROP_FRAME_HEIGHT, neededRes.height);
			cap.set(CAP_PROP_FRAME_WIDTH, neededRes.height * aR);
		}
		else {
			cap.set(CAP_PROP_FRAME_HEIGHT, neededRes.height);
			cap.set(CAP_PROP_FRAME_WIDTH, neededRes.width);
		}
		cap.set(CAP_PROP_AUTOFOCUS, 0);
	}
	else if (mCaptureParams.source == File && !cap.isOpened())
		cap.open(mCaptureParams.videoFileName);
	mImageSize = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));

	if (!cap.isOpened())
		throw std::runtime_error("Unable to open video source");

	Mat frame, processedFrame;
	while (cap.grab()) {
		cap.retrieve(frame);

		frame.copyTo(processedFrame);
		for (vector<Ptr<FrameProc>>::iterator it = processors.begin(); it != processors.end(); ++it)
			processedFrame = (*it)->processFrame(processedFrame);
		imshow(mainWindowName, processedFrame);
		char key = (char)waitKey(CAP_DELAY);

		if (key == 27) // esc
			return Finished;
		if (key == 114) // r
			return DeleteLastFrame;
		if (key == 100) // d
			return DeleteAllFrames;
		if (key == 115) // s
			return SaveCurrentData;
		if (key == 117) // u
			return SwitchUndistort;
		if (key == 118) // v
			return SwitchVisualisation;

		for (vector<Ptr<FrameProc>>::iterator it = processors.begin(); it != processors.end(); ++it)
			if ((*it)->isProcessed())
				return Calibrate;
	}
	return Finished;
}
Size CalibPipeline::getImageSize() const {
	return mImageSize;
}