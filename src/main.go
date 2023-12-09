package main

import (
	"fmt"
	"math/rand"
)

type Model struct {
	layers []Layer
}

type Layer struct {
	inputSize  int
	outputSize int
	weights    [][]float64
}

func main() {
	rand.Seed(1)
	model := createModel(5, 7, 3, 1)

	printModel(&model)

	inputs := []float64{2, 2, 3, 4, 5}
	result := forward(inputs, &model)
	fmt.Printf("Forward:\n  Inputs: %f\n   Result: %f", inputs, result)
}

func printModel(model *Model) {
	_layers := model.layers

	for i, _layer := range _layers {
		fmt.Printf("Layer %d: Input Size = %d, Output Size = %d\n", i, _layer.inputSize, _layer.outputSize)

		if len(_layer.weights) > 0 {
			fmt.Println("  Weights:")
			for _, row := range _layer.weights {
				fmt.Println("    ", row)
			}
		}
	}
}

func createModel(sizes ...int) Model {
	var _layers []Layer

	for i := 1; i < len(sizes); i++ {
		var _inputSize int = sizes[i-1]
		var _outputSize int = sizes[i]
		var _weights [][]float64

		for j := 0; j < _outputSize; j++ {
			_weight := make([]float64, _inputSize)
			_weights = append(_weights, _weight)

			for k := 0; k < _inputSize; k++ {
				_weights[j][k] = rand.Float64()
			}
		}

		layer := Layer{
			inputSize:  _inputSize,
			outputSize: _outputSize,
			weights:    _weights,
		}
		_layers = append(_layers, layer)
	}

	return Model{
		layers: _layers,
	}
}

func forward(inputs []float64, model *Model) []float64 {
	_current := inputs

	for _, _layer := range model.layers {
		var outputs []float64

		for _, node := range _layer.weights {
			var total float64 = 0

			for i, input := range _current {
				total += node[i] * input
			}
			outputs = append(outputs, total)
		}
		_current = outputs
	}

	return _current
}
