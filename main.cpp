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
using namespace cv::aruco;
using namespace std;
using namespace cv;


Ptr<CharucoBoard> makePrintBoard(CapParams cp);
void init();

int main() {
	init();
	CapParams capP;
	Ptr<CharucoBoard> charBoard = makePrintBoard(capP);
	InternalParams internalP;
	TermCriteria solverTCrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, internalP.solverMaxIters, internalP.solverEps);
	Ptr<CalibData> globalData(new CalibData);

	int calibFlags = 0;
	if (internalP.fastSolving) calibFlags |= CALIB_USE_QR;
	Ptr<CalibControl> cont(new CalibControl(globalData, calibFlags, true, capP.minFramesNum));
	Ptr<CalibDataControl> dataCont(new CalibDataControl(globalData, capP.maxFramesNum, internalP.alpha));
	dataCont->setParametersFileName("cameraParameters.xml");

	Ptr<FrameProc> capProc = Ptr<FrameProc>(new CalibProc(globalData, capP));
	Ptr<FrameProc> showProc = Ptr<FrameProc>(new ShowProc(globalData, cont));

	Ptr<CalibPipeline> pipeline(new CalibPipeline(capP));
	vector<Ptr<FrameProc>> procs;
	procs.push_back(capProc);
	procs.push_back(showProc);
	try {
		bool pipeFinBool = false;
		while (!pipeFinBool) {
			PipelineExitStatus exitStatus = pipeline->start(procs);
			if (exitStatus == Finished) {
				if (cont->getCommonCalibrationState())
					if ((static_cast<Ptr<CalibDataControl>*>(&dataCont))->get()->saveCurrentCameraParameters())
						cout << "Calibration parameters saved\n";
				pipeFinBool = true;
				continue;
			}
			if (exitStatus == Calibrate) {
				dataCont->rememberCurrentParameters();
				globalData->imgSz = pipeline->getImageSize();
				calibFlags = cont->getNewFlags();
				globalData->totalAvgErr = calibrateCameraCharuco(globalData->allCharCorns, globalData->allCharIds,
					charBoard, globalData->imgSz, globalData->camMat, globalData->distCos, noArray(),
					noArray(), globalData->stdDevs, noArray(), globalData->perViewErrors, calibFlags, solverTCrit);
				dataCont->updateUndistortMap();
				dataCont->printParametersToConsole(cout);
				cont->updateState();
				for (int j = 0; j < capP.calibStep; j++)
					dataCont->filterFrames();
				dynamic_cast<ShowProc*>(showProc.get())->updateBoardsView();
			}
			else if (exitStatus == DeleteLastFrame) {
				(static_cast<Ptr<CalibDataControl>*>(&dataCont))->get()->deleteLastFrame();
				cout << "Last frame deleted\n";
				dynamic_cast<ShowProc*>(showProc.get())->updateBoardsView();
			}
			else if (exitStatus == DeleteAllFrames) {
				(static_cast<Ptr<CalibDataControl>*>(&dataCont))->get()->deleteAllData();
				cout << "All frames deleted\n";
				dynamic_cast<ShowProc*>(showProc.get())->updateBoardsView();
			}
			else if (exitStatus == SaveCurrentData) {
				if ((static_cast<Ptr<CalibDataControl>*>(&dataCont))->get()->saveCurrentCameraParameters())
					cout << "Calibration parameters saved\n";
			}
			else if (exitStatus == SwitchUndistort)
				dynamic_cast<ShowProc*>(showProc.get())->switchUndistort();
			else if (exitStatus == SwitchVisualisation)
				dynamic_cast<ShowProc*>(showProc.get())->switchVisualizationMode();
			for (vector<Ptr<FrameProc>>::iterator it = procs.begin(); it != procs.end(); ++it)
				(*it)->resetState();
		}
	}
	catch (const runtime_error& exp) {
		cout << exp.what() << endl;
	}
	return 0;
}

Ptr<CharucoBoard> makePrintBoard(CapParams cp) {
	Ptr<Dictionary> dictionary = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(10));
	Ptr<CharucoBoard> charBoard = CharucoBoard::create(cp.boardSz.width, cp.boardSz.height, cp.sqrLen, cp.markLen, dictionary);
	cout << "sqrLen: " << cp.sqrLen << "     markLen: " << cp.markLen << "\n";
	Mat charImg;
	charBoard->draw(Size(700, 900), charImg, 30, 1);
	imwrite("charImg.png", charImg);
	return charBoard;
}
void init() {
	namedWindow(mainWindowName);
	moveWindow(mainWindowName, 10, 10);
	cout << consoleHelp;
}
bool calib::showOverlayMessage(const string& message) { cout << message; return false; }