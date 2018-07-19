#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "calibCommon.hpp"
#include "calibPipeline.hpp"
#include "FrameProc.hpp"
#include "calibControl.hpp"
#include "paramsControl.hpp"
using namespace calib;
using namespace aruco;
using namespace std;
using namespace cv;

Ptr<CharucoBoard> calib::makePrintBoard() {
	Ptr<Dictionary> dictionary = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(10));
	Ptr<CharucoBoard> charBoard = CharucoBoard::create(5, 5, .04f, .02f, dictionary);
	Mat charImg;
	charBoard->draw(Size(700, 900), charImg, 30, 1);
	imwrite("charImg.png", charImg);
	return charBoard;
}

int main() {
	namedWindow(mainWindowName);
	moveWindow(mainWindowName, 10, 10);
	cout << consoleHelp;
	Ptr<CharucoBoard> charBoard = makePrintBoard();
	ParamsControl paramsControl;
	if (!paramsControl.loadFromParser()) return 0;

	CapParams capParams = paramsControl.getCaptureParameters();
	InternalParams intParams = paramsControl.getInternalParameters();
	TermCriteria solverTermCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, intParams.solverMaxIters, intParams.solverEps);
	Ptr<CalibData> globalData(new CalibData);

	int calibFlags = 0;
	if (intParams.fastSolving) calibFlags |= CALIB_USE_QR;
	Ptr<CalibControl> controller(new CalibControl(globalData, calibFlags, true, capParams.minFramesNum));
	Ptr<CalibDataControl> dataControl(new CalibDataControl(globalData, capParams.maxFramesNum, intParams.alpha));
	dataControl->setParametersFileName("cameraParameters.xml");

	Ptr<FrameProc> capProcessor = Ptr<FrameProc>(new CalibProc(globalData, capParams));
	Ptr<FrameProc> showProcessor = Ptr<FrameProc>(new ShowProc(globalData, controller));

	Ptr<CalibPipeline> pipeline(new CalibPipeline(capParams));
	vector<Ptr<FrameProc>> processors;
	processors.push_back(capProcessor);
	processors.push_back(showProcessor);
	try {
		bool pipelineFinished = false;
		while (!pipelineFinished) {
			PipelineExitStatus exitStatus = pipeline->start(processors);
			if (exitStatus == Finished) {
				if (controller->getCommonCalibrationState())
					if ((static_cast<Ptr<CalibDataControl>*>(&dataControl))->get()->saveCurrentCameraParameters())
						cout << "Calibration parameters saved\n";
				pipelineFinished = true;
				continue;
			}
			if (exitStatus == Calibrate) {
				dataControl->rememberCurrentParameters();
				globalData->imgSz = pipeline->getImageSize();
				calibFlags = controller->getNewFlags();
				globalData->totalAvgErr = calibrateCameraCharuco(globalData->allCharCorns, globalData->allCharIds,
					charBoard, globalData->imgSz, globalData->camMat, globalData->distCos, noArray(),
					noArray(), globalData->stdDevs, noArray(), globalData->perViewErrors, calibFlags, solverTermCrit);
				dataControl->updateUndistortMap();
				dataControl->printParametersToConsole(cout);
				controller->updateState();
				for (int j = 0; j < capParams.calibStep; j++)
					dataControl->filterFrames();
				dynamic_cast<ShowProc*>(showProcessor.get())->updateBoardsView();
			}
			else if (exitStatus == DeleteLastFrame) {
				(static_cast<Ptr<CalibDataControl>*>(&dataControl))->get()->deleteLastFrame();
				cout << "Last frame deleted\n";
				dynamic_cast<ShowProc*>(showProcessor.get())->updateBoardsView();
			}
			else if (exitStatus == DeleteAllFrames) {
				(static_cast<Ptr<CalibDataControl>*>(&dataControl))->get()->deleteAllData();
				cout << "All frames deleted\n";
				dynamic_cast<ShowProc*>(showProcessor.get())->updateBoardsView();
			}
			else if (exitStatus == SaveCurrentData) {
				if ((static_cast<Ptr<CalibDataControl>*>(&dataControl))->get()->saveCurrentCameraParameters())
					cout << "Calibration parameters saved\n";
			}
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
bool calib::showOverlayMessage(const string& message) { cout << message; return false; }