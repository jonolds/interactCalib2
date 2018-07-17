#include "paramsControl.hpp"
#include <iostream>
using namespace std;
using namespace cv;

template <typename T>
static bool readFromNode(FileNode node, T& value) {
	if (!node.isNone()) {
		node >> value;
		return true;
	}
	return false;
}
static bool checkAssertion(bool value, const string& msg) {
	if (!value)
		cerr << "Error: " << msg << endl;
	return value;
}
bool calib::ParamsControl::loadFromFile(const string& inFile) {
	FileStorage reader;
	reader.open(inFile, FileStorage::READ);
	if (!reader.isOpened()) {
		cerr << "Can't open " << inFile << " -had default advanced params" << endl;
		return true;
	}
	readFromNode(reader["charuco_dict"], capParams.charucoDictName);
	readFromNode(reader["charuco_square_lenght"], capParams.charucoSquareLength);
	readFromNode(reader["charuco_marker_size"], capParams.charucoMarkerSize);
	readFromNode(reader["camera_resolution"], capParams.cameraResolution);
	readFromNode(reader["calibration_step"], capParams.calibrationStep);
	readFromNode(reader["max_frames_num"], capParams.maxFramesNum);
	readFromNode(reader["min_frames_num"], capParams.minFramesNum);
	readFromNode(reader["solver_eps"], mInternalParameters.solverEps);
	readFromNode(reader["solver_max_iters"], mInternalParameters.solverMaxIters);
	readFromNode(reader["fast_solver"], mInternalParameters.fastSolving);
	readFromNode(reader["frame_filter_conv_param"], mInternalParameters.filterAlpha);

	bool retValue =
		checkAssertion(capParams.charucoDictName >= 0, "Dict name must be >= 0") &&
		checkAssertion(capParams.charucoMarkerSize > 0, "Marker size must be positive") &&
		checkAssertion(capParams.charucoSquareLength > 0, "Square size must be positive") &&
		checkAssertion(capParams.minFramesNum > 1, "Minimal number of frames for calibration < 1") &&
		checkAssertion(capParams.calibrationStep > 0, "Calibration step must be positive") &&
		checkAssertion(capParams.maxFramesNum > capParams.minFramesNum, "maxFramesNum < minFramesNum") &&
		checkAssertion(mInternalParameters.solverEps > 0, "Solver precision must be positive") &&
		checkAssertion(mInternalParameters.solverMaxIters > 0, "Max solver iterations number must be positive") &&
		checkAssertion(mInternalParameters.filterAlpha >= 0 && mInternalParameters.filterAlpha <= 1,
		               "Frame filter convolution parameter must be in [0,1] interval") &&
		checkAssertion(capParams.cameraResolution.width > 0 && capParams.cameraResolution.height > 0,
		               "Wrong camera resolution values");
	reader.release();
	return retValue;
}

calib::ParamsControl::ParamsControl() {}
calib::CaptureParams calib::ParamsControl::getCaptureParameters() const {
	return capParams;
}
calib::InternalParams calib::ParamsControl::getInternalParameters() const {
	return mInternalParameters;
}
bool calib::ParamsControl::loadFromParser() {
	if (!checkAssertion(capParams.squareSize > 0, "Distance between corners or circles must be positive"))
		return false;
	if (!checkAssertion(capParams.templDst > 0, "Distance between parts of dual template must be positive"))
		return false;
	string templateType = "charuco";
	if (templateType.find("chessboard", 0) == 0) {
		capParams.board = Chessboard;
		capParams.boardSize = Size(7, 7);
	}
	else if (templateType.find("charuco", 0) == 0) {
		capParams.board = chAruco;
		capParams.boardSize = Size(5, 7);
		capParams.charucoDictName = 0;
		capParams.charucoSquareLength = 200;
		capParams.charucoMarkerSize = 100;
	}
	else {
		cerr << "Wrong template name\n";
		return false;
	}
	if (capParams.board == chAruco || capParams.board == Chessboard) {
		capParams.boardSize = Size(7, 7);
		if (!checkAssertion(capParams.boardSize.width > 0 || capParams.boardSize.height > 0,
		                    "Board size must be positive"))
			return false;
	}
	if (!checkAssertion(String("cameraParameters.xml").find(".xml") > 0, "Bad  outFile name: format is [name].xml"))
		return false;
	loadFromFile("defaultConfig.xml");
	return true;
}