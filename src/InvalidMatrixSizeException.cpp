#include "InvalidMatrixSizeException.h"

const char* InvalidMatrixSizeException::what() const throw() {
	return "Invalid matrix dimensions";
}