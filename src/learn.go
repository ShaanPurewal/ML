package main

func forward(inputs []float64, model *Model, log *[][]float64) []float64 {
	_current := inputs

	for _, _layer := range model.layers {
		var outputs []float64

		for i, node := range _layer.weights {
			outputs = append(outputs, calculateNode(node, _current)+_layer.biases[i])
		}

		*log = append(*log, _current)
		*log = append(*log, outputs)
		_current = activate(_layer.activation, outputs)
	}

	return _current
}

func calculateNode(node []float64, inputs []float64) float64 {
	var sum float64 = 0

	for i, input := range inputs {
		sum += node[i] * input
	}
	return sum
}

func back(inputs []float64, expected []float64, learnRate float64, model *Model) {
	var DCDA []float64
	var log [][]float64

	outputs := forward(inputs, model, &log)

	for i, _ := range outputs {
		DCDA = append(DCDA, costPrime(outputs[i], expected[i]))
	}

	for i := len(model.layers) - 1; i >= 0; i-- {
		activationPrimes := activatePrime(model.layers[i].activation, log[2*i+1])

		var newDCDA []float64 = make([]float64, model.layers[i].inputSize)
		for j, node := range model.layers[i].weights {
			DCDA[j] *= activationPrimes[j]
			for k, weight := range node {
				model.layers[i].weights[j][k] -= DCDA[j] * log[2*i][k] * learnRate
				newDCDA[k] += weight * DCDA[j]
			}

			model.layers[i].biases[j] -= DCDA[j] * learnRate
		}

		for j := 0; j < len(model.layers[i].weights[0]); j++ {
			newDCDA[j] /= float64(len(model.layers[i].weights[0]))
		}
		DCDA = newDCDA
	}
}
