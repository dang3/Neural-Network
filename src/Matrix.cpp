#include "Matrix.h"
#include "ActivationFunction.h"
#include <vector>
#include <iostream>

template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> m) {
	matrix = m;
}

template <typename T>
Matrix<T>::Matrix(int size) {
	matrix.resize(size);
}

template <typename T>
Matrix<T>::Matrix(){};

template <typename T>
void Matrix<T>::resize(int size) {
	matrix.resize(size);
}

template <typename T>
int Matrix<T>::getSizeAt(int row) {
	return matrix[row].size();
}

template <typename T>
int Matrix<T>::getSize() {
	return matrix.size();
}

template <typename T>
void Matrix<T>::insertAt(int row, T val) {
	matrix[row].push_back(val);
}

template <typename T>
double Matrix<T>::getValAt(int row, int col) {
	return matrix[row][col];
}

template <typename T>
Matrix<T> Matrix<T>::sigmoid(Matrix<T> input) {
	return ActivationFunction::sigmoid(input);
}

template <typename T>
void Matrix<T>::setValAt(int row, int col, T val) {
	matrix[row][col] = val;
}

template <typename T>
void Matrix<T>::resizeAt(int row, int size) {
	matrix[row].resize(size);
}

template <typename T>
Matrix<T> Matrix<T>::transpose(Matrix<T> input) {
	Matrix<T> result;
	result.resize(input.getSizeAt(0));

	for (int i = 0; i < input.getSizeAt(0); i++) {
		for (int j = 0; j < input.getSize(); j++) {
			result.insertAt(i, input.getValAt(j, i));
		}
	}
	return result;
}

template <typename T>
void Matrix<T>::printMatrix(Matrix<T> matrix) {
	MatrixOperations::printMatrix(matrix);
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &rhs) const {
	Matrix<T> m1 = this->matrix;
	Matrix<T> m2 = rhs.matrix;
	Matrix<T> output(m1.getSize());

	if (m1.getSizeAt(0) != m2.getSize()) { 	// check dimensions of matrices
		InvalidMatrixSizeException e;
		throw e;
	}

	for (int i = 0; i < m1.getSize(); i++) {
		for (int j = 0; j < m2.getSizeAt(0); j++) {
			double sum = 0;
			for (int k = 0; k < m1.getSizeAt(0); k++) {
				sum += m1.getValAt(i, k) * m2.getValAt(k, j);
			}
			output.insertAt(i, sum);
		}
	}
	return output;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &rhs) const {
	Matrix<T> m1 = this->matrix;
	Matrix<T> m2 = rhs.matrix;
	Matrix<T> result(m1.getSize());

	for (int i = 0; i < m1.getSize(); i++) {
		for (int j = 0; j < m1.getSizeAt(i); j++) {
			result.insertAt(i, m1.getValAt(i, j) - m2.getValAt(i, j));
		}
	}
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs) const {
	Matrix<T> m1 = this->matrix;
	Matrix<T> m2 = rhs.matrix;
	Matrix<T> result(m1.getSize());

	for (int i = 0; i < m1.getSize(); i++) {
		for (int j = 0; j < m1.getSizeAt(i); j++) {
			result.insertAt(i, m1.getValAt(i, j) + m2.getValAt(i, j));
		}
	}
	return result;
}
