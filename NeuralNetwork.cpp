#include "NeuralNetwork.h"
#include <iostream> 
#include <ctime>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include "Matrix.cpp"

using namespace std;

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
	this->inputNodes = inputNodes;
	this->hiddenNodes = hiddenNodes;
	this->outputNodes = outputNodes;
	this->learningRate = learningRate;
	srand(time(NULL));

	initialize();
	train();
	query();
}

void NeuralNetwork::initialize() {
	if (fileExists("w_in_hid.csv") && fileExists("w_hid_out.csv")) {
		cout << "Weight file exists, skipping training\n";
		w_in_hid.resize(hiddenNodes);
		w_hid_out.resize(outputNodes);
		readWeights();
	}
	else {
		cout << "Weight file not found, training\n";
		w_in_hid = initializeWeights(hiddenNodes, inputNodes);
		w_hid_out = initializeWeights(outputNodes, hiddenNodes);
	}

	// Setup dimensions of input Matrix
	input.resize(inputNodes);
	for (int i = 0; i < inputNodes; i++) {
		input.resizeAt(i, 1);
	}

	// Setup dimensions of all Matrices in hidden layer
	output_hidden.resize(hiddenNodes);
	error_hidden.resize(hiddenNodes);
	for (int i = 0; i < hiddenNodes; i++) {
		output_hidden.resizeAt(i, 1);
		error_hidden.resizeAt(i, 1);
	}

	// Setup dimensions of all output Matrices from final layer
	targetOut.resize(outputNodes);
	output.resize(outputNodes);
	error_output.resize(outputNodes); 
	for (int i = 0; i < outputNodes; i++) {
		targetOut.resizeAt(i, 1);
		output.resizeAt(i, 1);
		error_output.resizeAt(i, 1);
	}
}

Matrix<double> NeuralNetwork::initializeWeights(int rows, int cols) {
	srand(time(NULL));
	Matrix<double> weights(rows);
	int incomingLinks = cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			weights.insertAt(i, getRandNum(incomingLinks));
		}
	}
	return weights;
}

// returns a random number between -range to +range
double NeuralNetwork::getRandNum(double range) {
	double val = ((double)rand()) / ((double)RAND_MAX);	// random number between 0 to 1
	return (2 * val - 1) / sqrt(range);
}

void getValuesFromFile(string fileName) {
	fstream file(fileName, ios::in);
	
	file.close();
}

void NeuralNetwork::openFile(string fileName, void(NeuralNetwork::*fPtr)()) {
	fstream file(fileName, ios::in);
	int counter = 0;
	if (file) {
		string line;
		int counter = 0;
		int i = 0;
		while (getline(file, line)) {
			stringstream ss(line);
			string str;
			while (getline(ss, str, ',')) {
				double val = stof(str);
				if (counter++ % (inputNodes + 1) == 0) { // store every 785th char, discard everything
					if (i != 0)	{// begin process of weight update when input matrix is full
						(this->*fPtr)();
					}
 					resetTargetOut();
					targetOut.setValAt(val, 0, 0.99);
					i = 0;
				}
				else {
					input.setValAt(i++, 0, rescale(val));
				}

				if (counter%1000000 == 0)	// Print out every million, to see progress
					cout << counter << endl;
			}
		}
	}
	file.close();
}

void NeuralNetwork::train() {
	void(NeuralNetwork::*fPtr)() = &NeuralNetwork::updateWeights;
	openFile("mnist_train.csv", fPtr);

	// write weights to file
	recordWeights();
}

void NeuralNetwork::query() {
	void(NeuralNetwork::*fPtr)() = &NeuralNetwork::compareOutputs;
	openFile("mnist_test.csv", fPtr);

	cout << "num correct: " << numCorrect << endl;
	cout << "total labels: " << totalLabels << endl;
	cout << "Precision: " << (double)numCorrect / (double)totalLabels << endl;
}

void NeuralNetwork::updateWeights() {
	// calculate outputs
	forwardPropagate();

	// calculate errors
	error_output = targetOut - output;
	error_hidden = Matrix<double>::transpose(w_hid_out)*error_output;

	// update weights
	w_hid_out = w_hid_out + alpha_dE_by_dW(error_output, output, output_hidden);
	w_in_hid = w_in_hid + alpha_dE_by_dW(error_hidden, output_hidden, input);
}

void NeuralNetwork::compareOutputs() {
	totalLabels++;
	forwardPropagate();
	cout << "Actual output:" << endl;
	storeToFile(output);
	cout << "Target output:" << endl;
	storeToFile(targetOut);
	compareOutputMatrices();
}

void NeuralNetwork::compareOutputMatrices() {
	int maxIndexOutput = findMaxIndex(output);
	int maxIndexActual = findMaxIndex(targetOut);
	if (maxIndexOutput == maxIndexActual)
		numCorrect++;
}

int NeuralNetwork::findMaxIndex(Matrix<double> m) {
	int maxIndex = 0;
	double temp = 0;

	for (int i = 0; i < m.getSize(); i++) {
		if (temp < m.getValAt(i, 0))  {
			temp = m.getValAt(i, 0);
			maxIndex = i;
		}
	}
	return maxIndex;
}

double NeuralNetwork::rescale(int val) {
	return (val / 255.0)*0.99 + 0.01;
}

void NeuralNetwork::resetTargetOut() {
	for (int i = 0; i < outputNodes; i++) {
		targetOut.setValAt(i, 0, 0.01);
	}
}

void NeuralNetwork::storeToFile(Matrix<double> m) {
	fstream outputFile("output.txt", ios::app);

	for (int i = 0; i < m.getSize(); i++) {
		for (int j = 0; j < m.getSizeAt(i); j++) {
			cout << m.getValAt(i, j) << "   ";
			outputFile << m.getValAt(i, j) << "   ";
		}
		cout << endl;
		outputFile << endl;
	}
	cout << endl << endl;
	outputFile << endl << endl;

	outputFile.close();
}

Matrix<double> NeuralNetwork::alpha_dE_by_dW(Matrix<double> error_next, Matrix<double> output_next, Matrix<double> output_prev) {
	Matrix<double> m(error_next.getSize());

	for (int i = 0; i < error_next.getSize(); i++) {
		double sigVal = output_next.getValAt(i, 0) * (1 - output_next.getValAt(i, 0));
		m.insertAt(i, error_next.getValAt(i, 0)*sigVal*learningRate);
	}

	return m * Matrix<double>::transpose(output_prev);
}

void NeuralNetwork::forwardPropagate() {
	output_hidden = Matrix<double>::sigmoid(w_in_hid*input);
	output = Matrix<double>::sigmoid(w_hid_out * output_hidden);
}

void NeuralNetwork::recordWeights() {
	fstream w_in_hid_File("w_in_hid.csv", ios::out);
	fstream w_hid_out_File("w_hid_out.csv", ios::out);

	// write w_in_hid first
	if (w_in_hid_File) {
		for (int i = 0; i < w_in_hid.getSize(); i++) {
			for (int j = 0; j < w_in_hid.getSizeAt(0); j++) {
				w_in_hid_File << w_in_hid.getValAt(i, j) << ",";
			}
			w_in_hid_File << endl;
		}
	}
	w_in_hid_File.close();

	// write w_hid_out
	if (w_hid_out_File)	{
		for (int i = 0; i < w_hid_out.getSize(); i++) {
			for (int j = 0; j < w_hid_out.getSizeAt(0); j++) {
				w_hid_out_File << w_hid_out.getValAt(i, j) << ",";
			}
			w_hid_out_File << endl;
		}
	}
	w_hid_out_File.close();
}

void NeuralNetwork::readWeights() {
	// read inputs for w_in_hid
	fstream w_in_hid_File("w_in_hid.csv", ios::in);
	if (w_in_hid_File) {
		string line;
		int i = 0;
		while (getline(w_in_hid_File, line)) {
			stringstream ss(line);
			string str;
			while (getline(ss, str, ',')) {
				w_in_hid.insertAt(i, stof(str));
			}
			i++;
		}
	}
	w_in_hid_File.close();

	// read inputs for w_hid_out
	fstream w_hid_out_File("w_hid_out.csv", ios::in);
	if (w_hid_out_File) {
		string line;
		int i = 0;
		while (getline(w_hid_out_File, line)) {
			stringstream ss(line);
			string str;
			while (getline(ss, str, ',')) {
				w_hid_out.insertAt(i, stof(str));
			}
			i++;
		}
	}
	w_hid_out_File.close();
}

bool NeuralNetwork::fileExists(string fileName) {
	fstream file(fileName);
	return (bool)file;
}