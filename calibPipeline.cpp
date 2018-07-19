#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace calib;
using namespace std;
using namespace cv;

PipelineExitStatus CalibPipeline::start(vector<Ptr<FrameProc>> processors) {
	cap.open(camID);
	cap.set(CAP_PROP_FRAME_WIDTH, camRes.width);
	cap.set(CAP_PROP_FRAME_HEIGHT, camRes.height);
	cap.set(CAP_PROP_AUTOFOCUS, 0);
	mImageSize = Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
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