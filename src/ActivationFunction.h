#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <math.h>
#include <vector>
#include "Matrix.h"

class ActivationFunction {
public:
	static Matrix<double> sigmoid(Matrix<double> matrix) {
		for (int i = 0; i < matrix.getSize(); i++) {
			for (int j = 0; j < matrix.getSizeAt(i); j++) {
				matrix.setValAt(i,j, sigmoid(matrix.getValAt(i,j)));
			}
		}
		return matrix;
	}

	static double sigmoid(double x) {
		return 1 / (1 + exp(-1 * x));
	}

};

#endif