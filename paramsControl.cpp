#include "paramsControl.hpp"
#include <iostream>
using namespace std;
using namespace cv;
using namespace calib;

InternalParams ParamsControl::getInternalParameters() const {
	return mIntParams;
}