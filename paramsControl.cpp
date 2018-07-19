#include "paramsControl.hpp"
#include <iostream>
using namespace std;
using namespace cv;
using namespace calib;

template <typename T>
static bool readFile(FileNode node, T& val) { if (!node.isNone())node >> val; return (!node.isNone()); }
static bool chkAssert(bool val, const string& msg) { if (!val) cerr << "Error: " << msg; return val; }
bool ParamsControl::loadFromFile(const string& inFile) {


	FileStorage reader;
	reader.open(inFile, FileStorage::READ);
	if (!reader.isOpened()) { cerr << "Can't open " << inFile << "\n"; return true; }
	readFile(reader["charuco_dict"], capParams.charDictName);
	readFile(reader["charuco_square_lenght"], capParams.sqrLen);
	readFile(reader["charuco_marker_size"], capParams.charMarkSz);
	readFile(reader["camera_resolution"], capParams.camRes);
	readFile(reader["calibration_step"], capParams.calibStep);
	readFile(reader["max_frames_num"], capParams.maxFramesNum);
	readFile(reader["min_frames_num"], capParams.minFramesNum);
	readFile(reader["solver_eps"], mIntParams.solverEps);
	readFile(reader["solver_max_iters"], mIntParams.solverMaxIters);
	readFile(reader["fast_solver"], mIntParams.fastSolving);
	readFile(reader["frame_filter_conv_param"], mIntParams.alpha);
	bool retValue =
		chkAssert(capParams.charDictName >= 0, "Dict name must be >= 0\n") &&
		chkAssert(capParams.charMarkSz > 0, "Marker size must be positive\n") &&
		chkAssert(capParams.sqrLen > 0, "Square size must be positive\n") &&
		chkAssert(capParams.minFramesNum > 1, "Minimal number of frames for calibration < 1\n") &&
		chkAssert(capParams.calibStep > 0, "Calibration step must be positive\n") &&
		chkAssert(capParams.maxFramesNum > capParams.minFramesNum, "maxFramesNum < minFramesNum\n") &&
		chkAssert(mIntParams.solverEps > 0, "Solver precision must be positive\n") &&
		chkAssert(mIntParams.solverMaxIters > 0, "Max solver iterations number must be positive\n") &&
		chkAssert(mIntParams.alpha >= 0 && mIntParams.alpha <= 1, "alpha param must be in [0,1]\n") &&
		chkAssert(capParams.camRes.width > 0 && capParams.camRes.height > 0, "Wrong camera resolution values\n");
	reader.release();
	return retValue;
}

ParamsControl::ParamsControl() = default;
CapParams ParamsControl::getCaptureParameters() const { return capParams; }
InternalParams ParamsControl::getInternalParameters() const { return mIntParams; }
bool ParamsControl::loadFromParser() {
	capParams.boardSz = Size(5, 7);
	capParams.charDictName = 0;
	capParams.sqrLen = 200;
	capParams.charMarkSz = 100;
	return loadFromFile("defaultConfig.xml");
}