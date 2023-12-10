package main

import (
	"fmt"
)

const (
	defaultWeightRangeStart = -1
	defaultWeightRangeEnd   = 1

	defaultActivation       = "ReLU"
	defualtOutputActivation = "softmax"
)

func main() {
	defaultModel := createModel(defaultWeightRangeStart, defaultWeightRangeEnd, defaultActivation, defualtOutputActivation, 1, 5)

	manualLayers := []Layer{
		initLayer(1, 1, defaultWeightRangeStart, defaultWeightRangeEnd, "softmax"),
	}
	manualModel := Model{
		layers: manualLayers,
	}

	var verbose bool = true
	printModel(&defaultModel, verbose)
	printModel(&manualModel, verbose)

	inputs := []float64{1}
	resultDefault := forward(inputs, &defaultModel)
	resultManual := forward(inputs, &manualModel)

	fmt.Printf("Forward:\n  Inputs: %f\n   Result (Default): %f\n   Result (Manual): %f", inputs, resultDefault, resultManual)
}
