package collect4

type Layer struct {
	neurons []Neuron
}

func NewLayer(size int, input int) Layer {
	neurons := make([]Neuron, size)
	for i := range neurons {
		neurons[i] = NewNeuron(input)
	}
	return Layer{neurons}
}

func (l *Layer) Size() int {
	return len(l.neurons)
}

func (l *Layer) Compute(signal Vector) Vector {
	output := make([]float64, l.Size())
	for i, neuron := range l.neurons {
		output[i] = neuron.Compute(signal)
	}
	o := NewSimpleVector(output)
	o.Apply(activationFunction)
	return o
}