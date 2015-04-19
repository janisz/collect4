package connect4

import (
	"fmt"
	"math/rand"
)

type Perceptron struct {
	inputCount  int
	outputCount int
	layers      []*Layer
}

func NewPerceptron(in int, out int, layers []int) *Perceptron {
		p := &Perceptron{}
	p.inputCount = in
	p.outputCount = out
	hiddenLayersWithOutput := append(append([]int(nil), layers...), out)
	p.layers = make([]*Layer, len(hiddenLayersWithOutput))
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
	s += "OUT\n"
	for _, layer := range p.layers {
		s += layer.String() + "\n"
	}
	return s
}

func (p *Perceptron) Compute(input Vector) Vector {
	return p.ComputeVerbose(input)[len(p.layers)-1]
}

func (p *Perceptron) ComputeVerbose(input Vector) []Vector {
	outputs := make([]Vector, len(p.layers) )
	log.Debug("Computing...")
	signal := input
	for i, layer := range p.layers {
		signal = layer.Compute(signal)
		outputs[i] = signal
		log.Debug("Layer %d output = %f", i, signal)
	}
	return outputs
}


func (p *Perceptron) Learn(coach Coach, maxError float64, iterations int) {
	for i:=0; i<iterations; i++ {
		coach.Iteration()
		error := coach.Error()
		log.Debug("Error %f", error)
		if (error < maxError) {
			log.Info("Stop training after %d with error %f < %f",  i, error, maxError)
			break
		}
	}
	log.Info("Training finished")
}

func (p *Perceptron) Initialize(seed int64) {
	rand.Seed(seed)
	for _, layer := range p.layers {
		for _, neuron := range layer.neurons {
			neuron.weights = NewZeroVector(neuron.weights.Length())
			neuron.weights.Apply(func (x float64) float64 {
				return rand.Float64() - 0.5
			})
		}
	}
}
