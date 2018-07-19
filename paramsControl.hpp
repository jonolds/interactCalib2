#ifndef PARAMETERS_CONTROLLER_HPP
#define PARAMETERS_CONTROLLER_HPP
#include <string>
#include <opencv2/core.hpp>
#include "calibCommon.hpp"
namespace calib {
	class ParamsControl {
	protected:
		CapParams capParams;
		InternalParams mIntParams;
	public:
		bool loadFromFile(const std::string& inFile);
		ParamsControl();
		CapParams getCaptureParameters() const;
		InternalParams getInternalParameters() const;
		bool loadFromParser();
	};
}
#endif