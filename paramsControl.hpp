#ifndef PARAMETERS_CONTROLLER_HPP
#define PARAMETERS_CONTROLLER_HPP
#include <string>
#include <opencv2/core.hpp>
#include "calibCommon.hpp"

namespace calib {
	class ParamsControl {
	protected:
		InternalParams mIntParams;
	public:
		ParamsControl() = default;
		InternalParams getInternalParameters() const;
	};
}
#endif