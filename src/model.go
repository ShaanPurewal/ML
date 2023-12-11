package main

import (
	"fmt"
	"math/rand"
)

func printModel(model *Model, verbose bool) {
	_layers := model.layers
	spacing := "  "

	fmt.Println("Model:")
	for i, _layer := range _layers {
		fmt.Printf("%sLayer %d: Input Size = %d, Output Size = %d, Activation = %s\n", spacing, i, _layer.inputSize, _layer.outputSize, _layer.activation)

		if len(_layer.weights) > 0 && verbose {
			fmt.Println(spacing, spacing, "Weights:")
			for j, weight := range _layer.weights {
				fmt.Println(spacing, spacing, spacing, weight, " Bias: ", _layer.biases[j])
			}
		}
	}
}

func createModel(weightRangeStart float64, weightRangeEnd float64, activation string, outputActivation string, shape ...int) Model {
	var _layers []Layer

	for i := 1; i < len(shape); i++ {
		_activation := activation
		if i == len(shape)-1 {
			_activation = outputActivation
		}

		_layers = append(_layers, initLayer(shape[i-1], shape[i], weightRangeStart, weightRangeEnd, _activation))
	}

	return Model{
		layers: _layers,
	}
}

func initLayer(inputSize int, outputSize int, weightRangeStart float64, weightRangeEnd float64, activation string) Layer {
	var _weights [][]float64
	var _biases []float64

	for j := 0; j < outputSize; j++ {
		_weight := make([]float64, inputSize)
		_weights = append(_weights, _weight)

		for k := 0; k < inputSize; k++ {
			_weights[j][k] = (weightRangeEnd-weightRangeStart)*rand.Float64() + weightRangeStart
		}

		_biases = append(_biases, (weightRangeEnd-weightRangeStart)*rand.Float64()-weightRangeEnd)
	}

	return Layer{
		inputSize:  inputSize,
		outputSize: outputSize,
		weights:    _weights,
		biases:     _biases,
		activation: activation,
	}
}
