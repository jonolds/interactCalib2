#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <stdexcept>

using namespace calib;
using namespace std;
using namespace cv;

Size CalibPipeline::getCameraResolution() {
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	int w = int(cap.get(CAP_PROP_FRAME_WIDTH)), h = int(cap.get(CAP_PROP_FRAME_HEIGHT));
	return Size(w, h);
}
CalibPipeline::CalibPipeline(CapParams params) : capParams(move(params)) {}

PipelineExitStatus CalibPipeline::start(vector<Ptr<FrameProc>> processors) {
	cap.open(capParams.camID);
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(CAP_PROP_AUTOFOCUS, 0);
	mImageSize = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
	if (!cap.isOpened())
		throw runtime_error("Unable to open video source");
	Mat frame, processedFrame;
	while (cap.grab()) {
		cap.retrieve(frame);
		frame.copyTo(processedFrame);
		for (vector<Ptr<FrameProc>>::iterator it = processors.begin(); it != processors.end(); ++it)
			processedFrame = (*it)->processFrame(processedFrame);
		imshow(mainWindowName, processedFrame);
		char key = (char)waitKey(10);
		if (key == 27) /*esc*/ return Finished;
		if (key == 114) /*r*/  return DeleteLastFrame;
		if (key == 100) /*d*/  return DeleteAllFrames;
		if (key == 115) /*s*/  return SaveCurrentData;
		if (key == 117) /*u*/  return SwitchUndistort;
		if (key == 118) /*v*/  return SwitchVisualisation;
		for (vector<Ptr<FrameProc>>::iterator it = processors.begin(); it != processors.end(); ++it)
			if ((*it)->isProcessed())
				return Calibrate;
	}
	return Finished;
}
Size CalibPipeline::getImageSize() const {
	return mImageSize;
}