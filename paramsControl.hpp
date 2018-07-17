#ifndef PARAMETERS_CONTROLLER_HPP
#define PARAMETERS_CONTROLLER_HPP
#include <string>
#include <opencv2/core.hpp>
#include "calibCommon.hpp"
namespace calib {
	class ParamsControl {
	protected:
		CaptureParams capParams;
		InternalParams mInternalParameters;
	public:
		bool loadFromFile(const std::string& inFile);
		ParamsControl();
		CaptureParams getCaptureParameters() const;
		InternalParams getInternalParameters() const;
		bool loadFromParser();
	};
}
#endif