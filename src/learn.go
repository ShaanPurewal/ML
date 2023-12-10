package main

func forward(inputs []float64, model *Model) []float64 {
	_current := inputs

	for _, _layer := range model.layers {
		var outputs []float64

		for _, node := range _layer.weights {
			outputs = append(outputs, calculateNode(node, _current))
		}
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
