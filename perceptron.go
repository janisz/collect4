package collect4

import (
	"fmt"
	"math/rand"
	"github.com/op/go-logging"
	"os"
)

type Perceptron struct {
	inputCount  int
	outputCount int
	layers      []*Layer
}

var log = logging.MustGetLogger("example")
var format = logging.MustStringFormatter(
"%{color}%{time:15:04:05.000} %{shortfunc} ▶ %{level:.4s} %{id:03x}%{color:reset} %{message}",
)

func NewPerceptron(in int, out int, layers []int) *Perceptron {

	// For demo purposes, create two backend for os.Stderr.
	backend1 := logging.NewLogBackend(os.Stderr, "", 0)
	backend2 := logging.NewLogBackend(os.Stderr, "", 0)

	// For messages written to backend2 we want to add some additional
	// information to the output, including the used log level and the name of
	// the function.
	backend2Formatter := logging.NewBackendFormatter(backend2, format)

	// Only errors and more severe messages should be sent to backend1
	backend1Leveled := logging.AddModuleLevel(backend1)
	backend1Leveled.SetLevel(logging.ERROR, "")

	// Set the backends to be used.
	logging.SetBackend(backend1Leveled, backend2Formatter)

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
	log.Info("Computing...")
	signal := input
	for i, layer := range p.layers {
		signal = layer.Compute(signal)
		outputs[i] = signal
		log.Debug("Layer %d output = %f", i, signal)
	}
	return outputs
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
