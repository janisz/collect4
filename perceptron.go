package collect4

import (
	"math"
	"fmt"
)

type Perceptron struct {
	inputCount  int
	outputCount int
	layers      []Layer
}

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
		output [i] = neuron.Compute(signal)
	}
	o := NewSimpleVector(output)
	o.Apply(math.Tanh)
	return o
}

type Neuron struct {
	weights Vector
}

func NewNeuron(input int) Neuron {
	return Neuron{
		weights: NewSimpleVector(make([]float64, input)),
	}
}

func (n *Neuron) Compute(signal Vector) float64 {
	output := signal.Copy()
	output.MulElements(n.weights)
	return output.Sum()
}

func NewPerceptron(in int, out int, layers []int) *Perceptron {
	p := &Perceptron{}
	p.inputCount = in
	p.outputCount = out
	hiddenLayersWithOutput := append(append([]int(nil), layers...), out)
	p.layers = make([]Layer, len(hiddenLayersWithOutput))
	for i := range p.layers {
		var input int
		if (i == 0) {
			input = in
		} else {
			input = hiddenLayersWithOutput[i-1]
		}
		p.layers[i] = NewLayer(hiddenLayersWithOutput[i], input)
	}

	return p
}

func (p *Perceptron) String() string {
	s := fmt.Sprintf("%d->", p.inputCount)
	for _, layer := range p.layers {
		s += fmt.Sprintf("%d->", layer.Size())
	}
	s += "OUT"
	return s
}

func (p *Perceptron) Compute(input Vector) Vector {
	signal := input
	for _, layer := range p.layers {
		signal = layer.Compute(signal)
	}
	return signal
}
