#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include "InvalidMatrixSizeException.h"

template <typename T>
class Matrix {
private:
	std::vector < std::vector<T> > matrix;

public:
	Matrix<T>(std::vector < std::vector<T> >);
	Matrix<T>(int);
	Matrix<T>();

	void resize(int);
	void resizeAt(int, int);
	int getSize();
	int getSizeAt(int);

	double getValAt(int, int);
	void setValAt(int row, int col, T val);	// equivalent to matrix[row][col] = val
	void insertAt(int row, T val);	// equivalent to matrix[row].push_back(val)

	friend class NeuralNetwork;
	
	static Matrix<T> sigmoid(Matrix<T> input);
	static Matrix<T> transpose(Matrix<T> input);
	static void printMatrix(Matrix<T> m);

	Matrix<T> operator*(const Matrix<T> &rhs) const;
	Matrix<T> operator-(const Matrix<T> &rhs) const;
	Matrix<T> operator+(const Matrix<T> &rhs) const;

};
#endif