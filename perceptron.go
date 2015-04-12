package collect4

import (
	"fmt"
	"math/rand"
)

type Perceptron struct {
	inputCount  int
	outputCount int
	layers      []Layer
}

func NewPerceptron(in int, out int, layers []int) *Perceptron {
	p := &Perceptron{}
	p.inputCount = in
	p.outputCount = out
	hiddenLayersWithOutput := append(append([]int(nil), layers...), out)
	p.layers = make([]Layer, len(hiddenLayersWithOutput))
	for i := range p.layers {
		var input int
		if i == 0 {
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

func (p *Perceptron) Learn(coach Coach, iterations int) {
	for i:=0; i<iterations; i++ {
		coach.Iteration()
	}
}

func (p *Perceptron) Initialize(seed int64) {
	rand.Seed(seed)
	for _, layer := range p.layers {
		for i, neuron := range layer.neurons {
			weights := make([]float64, neuron.weights.Length())
			for i := range weights {
				weights[i] = rand.Float64() - rand.Float64()
			}
			layer.neurons[i].weights = NewSimpleVector(weights)
		}
	}
}
