#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include "Matrix.h"

class NeuralNetwork {
private:
	int layers = 3, inputNodes, hiddenNodes, outputNodes;
	int numCorrect, totalLabels;
	double learningRate;
	Matrix<double> w_in_hid;	// dimensions: hiddenNotes x inputNodes
	Matrix<double> w_hid_out;	// dimensions: outputNodes x hiddenNodes
	Matrix<double> input, targetOut;
	Matrix<double> output_hidden, output;
	Matrix<double> error_hidden, error_output;


	void resetTargetOut();
	double getRandNum(double);
	Matrix<double> alpha_dE_by_dW(Matrix<double> error, Matrix<double> weight, Matrix<double> output);

	double rescale(int);
	void updateWeights();
	void compareOutputs();
	void storeToFile(Matrix<double> m);
	void recordWeights();
	void readWeights();
	bool fileExists(std::string);

	void forwardPropagate();

	void openFile(std::string, void(NeuralNetwork::*fPtr)());
	void compareOutputMatrices();
	int findMaxIndex(Matrix<double>);

	Matrix<double> initializeWeights(int, int);	// generates a matrix of random weights

public:
	NeuralNetwork(int, int, int, double);
	void initialize();
	void train();
	void query();
};

#endif