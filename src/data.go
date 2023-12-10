package main

import (
	"math"
	"strings"
)

type Model struct {
	layers []Layer
}

type Layer struct {
	inputSize  int
	outputSize int

	weights [][]float64

	activation string
}

func activate(activationFunction string, values []float64) []float64 {
	var results []float64
	var sum float64 = 0

	for _, value := range values {
		switch strings.ToLower(activationFunction) {
		case "relu":
			results = append(results, ReLU(value))
		case "sigmoid":
			results = append(results, sigmoid(value))
		case "tanh":
			results = append(results, tanh(value))
		case "softmax":
			sum += math.Exp(value)
			results = append(results, math.Exp(value))
		default:
			results = append(results, value)
		}
	}

	if strings.EqualFold(activationFunction, "softmax") {
		for i, _ := range results {
			results[i] /= sum
		}
	}

	return results
}

func ReLU(value float64) float64 {
	if value <= 0 {
		return 0
	}
	return value
}

func sigmoid(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

func tanh(value float64) float64 {
	return math.Tanh(value)
}
