#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "calibCommon.hpp"
#include "calibPipeline.hpp"
#include "FrameProc.hpp"
#include "calibControl.hpp"
#include "paramsControl.hpp"
#include "rotationConverters.hpp"
using namespace calib;
using namespace std;
using namespace cv;

const string keys =
	"{sz       | 16.3    | Distance between two nearest centers of circles or squares on calibration board}"
	"{ft       | true    | Auto tuning of calibration flags}"
	"{vis      | grid    | Captured boards visualisation (grid, window)}"
	"{pf       | defaultConfig.xml| Advanced application parameters}";


bool calib::showOverlayMessage(const string& message) {
	cout << message << "\n";
	return false;
}
static void deleteButton(int, void* data) {
	(static_cast<Ptr<CalibDataControl>*>(data))->get()->deleteLastFrame();
	cout << "Last frame deleted\n";
}
static void deleteAllButton(int, void* data) {
	(static_cast<Ptr<CalibDataControl>*>(data))->get()->deleteAllData();
	cout << "All frames deleted\n";
}
static void saveCurrentParamsButton(int, void* data) {
	if ((static_cast<Ptr<CalibDataControl>*>(data))->get()->saveCurrentCameraParameters())
		cout << "Calibration parameters saved\n";
}

int main(int argc, char** argv) {
	CommandLineParser parser(argc, argv, keys);
	cout << consoleHelp << "\n";
	ParamsControl paramsControl;

	if (!paramsControl.loadFromParser()) return 0;

	CaptureParams capParams = paramsControl.getCaptureParameters();
	InternalParams intParams = paramsControl.getInternalParameters();
	TermCriteria solverTermCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
	                                           intParams.solverMaxIters, intParams.solverEps);
	Ptr<CalibData> globalData(new CalibData);
	//if (!parser.has("v")) globalData->imageSize = capParams.cameraResolution;

	int calibFlags = 0;
	if (intParams.fastSolving) calibFlags |= CALIB_USE_QR;
	Ptr<CalibControl> controller(new CalibControl(globalData, calibFlags, true, capParams.minFramesNum));
	Ptr<CalibDataControl> dataControl(new CalibDataControl(globalData, capParams.maxFramesNum, intParams.filterAlpha));
	dataControl->setParametersFileName("cameraParameters.xml");

	Ptr<FrameProc> showProcessor, capProcessor = cv::Ptr<FrameProc>(new CalibProc(globalData, capParams));
	showProcessor = Ptr<FrameProc>(new ShowProc(globalData, controller, capParams.board));

	if (parser.get<string>("vis").find("window") == 0) {
		dynamic_cast<ShowProc*>(showProcessor.get())->setVisualizationMode(Window);
		namedWindow(gridWindowName);
		moveWindow(gridWindowName, 1280, 500);
	}

	Ptr<CalibPipeline> pipeline(new CalibPipeline(capParams));
	vector<Ptr<FrameProc>> processors;
	processors.push_back(capProcessor);
	processors.push_back(showProcessor);

	namedWindow(mainWindowName);
	moveWindow(mainWindowName, 10, 10);
	try {
		bool pipelineFinished = false;
		while (!pipelineFinished) {
			PipelineExitStatus exitStatus = pipeline->start(processors);
			if (exitStatus == Finished) {
				if (controller->getCommonCalibrationState())
					saveCurrentParamsButton(0, &dataControl);
				pipelineFinished = true;
				continue;
			}
			if (exitStatus == Calibrate) {
				dataControl->rememberCurrentParameters();
				globalData->imageSize = pipeline->getImageSize();
				calibFlags = controller->getNewFlags();
				if (capParams.board != chAruco) {
					globalData->totalAvgErr = calibrateCamera(globalData->objectPoints, globalData->imagePoints,
						globalData->imageSize, globalData->cameraMatrix, globalData->distCoeffs, noArray(), 
						noArray(), globalData->stdDeviations, noArray(), globalData->perViewErrors, calibFlags, solverTermCrit);
				}
				else {
#ifdef HAVE_OPENCV_ARUCO
					Ptr<aruco::Dictionary> dictionary =
						getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(capParams.charucoDictName));
					Ptr<aruco::CharucoBoard> charucoboard =
						aruco::CharucoBoard::create(capParams.boardSize.width, capParams.boardSize.height,
						                            capParams.charucoSquareLength, (float)capParams.charucoMarkerSize, dictionary);
					globalData->totalAvgErr = calibrateCameraCharuco(globalData->allCharucoCorners, globalData->allCharucoIds,
						charucoboard, globalData->imageSize,	globalData->cameraMatrix, globalData->distCoeffs, noArray(), 
						noArray(), globalData->stdDeviations, noArray(), globalData->perViewErrors, calibFlags, solverTermCrit);
#endif
				}
				dataControl->updateUndistortMap();
				dataControl->printParametersToConsole(cout);
				controller->updateState();
				for (int j = 0; j < capParams.calibrationStep; j++)
					dataControl->filterFrames();
				dynamic_cast<ShowProc*>(showProcessor.get())->updateBoardsView();
			}
			else if (exitStatus == DeleteLastFrame) {
				deleteButton(0, &dataControl);
				dynamic_cast<ShowProc*>(showProcessor.get())->updateBoardsView();
			}
			else if (exitStatus == DeleteAllFrames) {
				deleteAllButton(0, &dataControl);
				dynamic_cast<ShowProc*>(showProcessor.get())->updateBoardsView();
			}
			else if (exitStatus == SaveCurrentData)
				saveCurrentParamsButton(0, &dataControl);
			else if (exitStatus == SwitchUndistort)
				dynamic_cast<ShowProc*>(showProcessor.get())->switchUndistort();
			else if (exitStatus == SwitchVisualisation)
				dynamic_cast<ShowProc*>(showProcessor.get())->switchVisualizationMode();
			for (vector<Ptr<FrameProc>>::iterator it = processors.begin(); it != processors.end(); ++it)
				(*it)->resetState();
		}
	}
	catch (const runtime_error& exp) {
		cout << exp.what() << endl;
	}
	return 0;
}