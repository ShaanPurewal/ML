package main

type Model struct {
	layers []Layer
}

type Layer struct {
	inputSize  int
	outputSize int

	weights [][]float64

	activation string
}

func activate(activationFunction string, value float64) float64 {
	switch activationFunction {
	case "ReLU":
		return ReLU(value)
	default:
		return value
	}
}

func ReLU(value float64) float64 {
	if value <= 0 {
		return 0
	}
	return value
}
