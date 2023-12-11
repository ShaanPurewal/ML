package main

import (
	"fmt"
	"math/rand"
)

const (
	defaultWeightRangeStart = -10
	defaultWeightRangeEnd   = 10

	defaultActivation       = "ReLU"
	defualtOutputActivation = "softmax"
)

func main() {
	rand.Seed(1)
	//defaultModel := createModel(defaultWeightRangeStart, defaultWeightRangeEnd, defaultActivation, defualtOutputActivation, 1, 5)

	manualLayers := []Layer{
		initLayer(2, 200, defaultWeightRangeStart, defaultWeightRangeEnd, "sigmoid"),
		initLayer(200, 100, defaultWeightRangeStart, defaultWeightRangeEnd, "sigmoid"),
		initLayer(100, 50, defaultWeightRangeStart, defaultWeightRangeEnd, "sigmoid"),
		initLayer(50, 1, defaultWeightRangeStart, defaultWeightRangeEnd, ""),
	}
	manualModel := Model{
		layers: manualLayers,
	}

	var verbose bool = true
	//printModel(&defaultModel, verbose)
	printModel(&manualModel, verbose)

	inputs := []float64{1, 5}

	learnRate := 0.0001

	resultDefault := 0.0 //forward(inputs, &defaultModel)

	var log [][]float64
	resultManual := forward(inputs, &manualModel, &log)

	fmt.Printf("Forward:\n  Inputs: %f\n   Result (Default): %f\n   Result (Manual): %f\n", inputs, resultDefault, resultManual)

	last := 100.0
	n := 1.0
	for i := 0; i < 10000000 && !(last/n < 30 && n > 1000); i++ {
		input := []float64{rand.Float64() * 10, rand.Float64() * 10}

		back(input, expected(input), learnRate*(n/1000), &manualModel)

		n += 1
		c := cost(forward(input, &manualModel, &log), expected(input))
		last += c

		if i%1000 == 0 {
			fmt.Println(last / n)
		}
	}

	//printModel(&manualModel, verbose)
	fmt.Println(forward([]float64{1, 1}, &manualModel, &log))
	fmt.Println(forward([]float64{10, 1}, &manualModel, &log))
	fmt.Println(forward([]float64{1, 10}, &manualModel, &log))
	fmt.Println(forward([]float64{5, 5}, &manualModel, &log))
	fmt.Println(forward([]float64{10, 10}, &manualModel, &log))

}

func expected(input []float64) []float64 {
	return []float64{input[0] * input[1]}
}
