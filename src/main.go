package main

import (
	"fmt"
)

const (
	defaultWeightRangeStart = -1
	defaultWeightRangeEnd   = 1

	defaultActivation = "ReLU"
)

func main() {
	defaultModel := createModel(defaultWeightRangeStart, defaultWeightRangeEnd, defaultActivation, 1, 5, 5, 1)

	manualLayers := []Layer{
		initLayer(1, 5, defaultWeightRangeStart, defaultWeightRangeEnd, defaultActivation),
		initLayer(5, 5, defaultWeightRangeStart, defaultWeightRangeEnd, defaultActivation),
		initLayer(5, 1, defaultWeightRangeStart, defaultWeightRangeEnd, defaultActivation),
	}
	manualModel := Model{
		layers: manualLayers,
	}

	var verbose bool = false
	printModel(&defaultModel, verbose)
	printModel(&manualModel, verbose)

	inputs := []float64{1}
	resultDefault := forward(inputs, &defaultModel)
	resultManual := forward(inputs, &manualModel)

	fmt.Printf("Forward:\n  Inputs: %f\n   Result (Default): %f\n   Result (Manual): %f", inputs, resultDefault, resultManual)
}
