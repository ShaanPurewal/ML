package main

func forward(inputs []float64, model *Model) []float64 {
	_current := inputs

	for _, _layer := range model.layers {
		var outputs []float64

		for _, node := range _layer.weights {
			var result float64 = calculateNode(node, _current)
			outputs = append(outputs, activate(_layer.activation, result))
		}
		_current = outputs
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
