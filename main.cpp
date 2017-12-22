#include <iostream>
#include "NeuralNetwork.h"
#include <vector>
#include <ctime>

using namespace std;

int main() {

	const int inputNodes = 784;
	const int hiddenNodes = 100;
	const int outputNodes = 10;
	const double learningRate = 0.3;

	NeuralNetwork network(inputNodes, hiddenNodes, outputNodes, learningRate);

	cin.get();

}
