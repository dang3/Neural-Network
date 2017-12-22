#ifndef INVALIDMATRIXSIZEEXCEPTION_H
#define INVALIDMATRIXSIZEEXCEPTION_H

#include <string>

class InvalidMatrixSizeException : public std::exception {
public:
	const char* what() const throw();
};

#endif