package perceptron

import (
	"math"
	"math/rand"
)

type ActivationFunction int

const (
	SIGMOID = iota
	TANH
)

type Perceptron struct {
	Bias               bool
	Sizes              []int
	Thresholds         [][]float64
	outputs            [][]float64
	errors             [][]float64
	deltas             [][]float64
	Weights            [][][]float64
	changes            [][][]float64
	ActivationFunction ActivationFunction
}

func NewPerceptron(sizes []int, bias bool, activation ActivationFunction) Perceptron {
	return Perceptron{
		Bias:               bias,
		Sizes:              sizes,
		ActivationFunction: activation,
	}
}

//This function must be called after unmarshalling perceptron
func (p *Perceptron) SetUp() {

	rand.Seed(1024)

	p.changes = make([][][]float64, len(p.Sizes))

	p.outputs = make([][]float64, len(p.Sizes))
	p.errors = make([][]float64, len(p.Sizes))
	p.deltas = make([][]float64, len(p.Sizes))

	for i := range p.Weights {

		p.outputs[i] = make([]float64, p.Sizes[i])
		p.errors[i] = make([]float64, p.Sizes[i])
		p.deltas[i] = make([]float64, p.Sizes[i])

		p.changes[i] = make([][]float64, p.Sizes[i])

		for j := 0; j < p.Sizes[i]; j++ {
			if i == 0 {
				p.changes[i][j] = make([]float64, 0)
			} else {
				p.changes[i][j] = make([]float64, p.Sizes[i-1])
			}

		}
	}
}

func (p *Perceptron) Initialize() {

	rand.Seed(1024)

	p.Weights = make([][][]float64, len(p.Sizes))
	p.changes = make([][][]float64, len(p.Sizes))

	p.Thresholds = make([][]float64, len(p.Sizes))
	p.outputs = make([][]float64, len(p.Sizes))
	p.errors = make([][]float64, len(p.Sizes))
	p.deltas = make([][]float64, len(p.Sizes))

	for i := range p.Weights {

		p.Thresholds[i] = make([]float64, p.Sizes[i])
		p.outputs[i] = make([]float64, p.Sizes[i])
		p.errors[i] = make([]float64, p.Sizes[i])
		p.deltas[i] = make([]float64, p.Sizes[i])

		p.Weights[i] = make([][]float64, p.Sizes[i])
		p.changes[i] = make([][]float64, p.Sizes[i])

		for j := 0; j < p.Sizes[i]; j++ {
			if i == 0 {
				p.Weights[i][j] = randoms(0)
				p.changes[i][j] = make([]float64, 0)
			} else {
				p.Weights[i][j] = randoms(p.Sizes[i-1])
				p.changes[i][j] = make([]float64, p.Sizes[i-1])
			}

		}
	}
}

func (p *Perceptron) outputLayer() int {
	return len(p.Sizes) - 1
}

func (p *Perceptron) Compute(input []float64) []float64 {
	p.outputs[0] = input
	for layer := 1; layer <= p.outputLayer(); layer++ {
		for neuron := 0; neuron < p.Sizes[layer]; neuron++ {
			neuronWeights := p.Weights[layer][neuron]
			sum := 0.0
			if p.Bias {
				sum += p.Thresholds[layer][neuron]
			}
			for k := range neuronWeights {
				sum += neuronWeights[k] * input[k]
			}
			p.outputs[layer][neuron] = p.activationFunction(sum)
		}
		input = p.outputs[layer]
	}
	return p.outputs[p.outputLayer()]
}

func (p *Perceptron) Learn(input, ideal [][]float64, learningRate, momentum float64, iterations int, errorThreshold float64) (float64, int) {
	globalError := 1.0
	iteration := 0
	for iteration = 0; iteration < iterations && globalError > errorThreshold; iteration++ {
		errorSum := 0.0
		for example := range input {
			p.exercise(input[example], ideal[example], learningRate, momentum)
			errorSum += MeanSquaredError(p.errors[p.outputLayer()])
		}
		globalError = errorSum / float64(len(input))
	}
	return globalError, iteration
}

func (p *Perceptron) exercise(input, ideal []float64, learningRate, momentum float64) {
	p.Compute(input)
	p.computeDeltas(ideal)
	p.updateWeights(learningRate, momentum)
}

func (p *Perceptron) computeDeltas(ideal []float64) {
	for layer := p.outputLayer(); layer >= 0; layer-- {
		for neuron := 0; neuron < p.Sizes[layer]; neuron++ {
			output := p.outputs[layer][neuron]
			error := 0.0
			if layer == p.outputLayer() {
				error = ideal[neuron] - output
			} else {
				deltas := p.deltas[layer+1]
				for i, delta := range deltas {
					error += delta * p.Weights[layer+1][i][neuron]
				}
			}
			p.errors[layer][neuron] = error
			p.deltas[layer][neuron] = error * p.activationFunctionDerivative(output)
		}
	}
}

func (p *Perceptron) activationFunction(x float64) float64 {
	function, _ := activationFunction(p.ActivationFunction)
	return function(x)
}

func (p *Perceptron) activationFunctionDerivative(x float64) float64 {
	_, derivative := activationFunction(p.ActivationFunction)
	return derivative(x)
}

func (p *Perceptron) updateWeights(learningRate float64, momentum float64) {
	for layer := 1; layer <= p.outputLayer(); layer++ {
		incoming := p.outputs[layer-1]

		for neuron := 0; neuron < p.Sizes[layer]; neuron++ {
			delta := p.deltas[layer][neuron]

			for previousLayerNeuron := range incoming {
				change := p.changes[layer][neuron][previousLayerNeuron]

				change = learningRate*delta*incoming[previousLayerNeuron] + momentum*change

				p.changes[layer][neuron][previousLayerNeuron] = change
				p.Weights[layer][neuron][previousLayerNeuron] += change
			}
			if p.Bias {
				p.Thresholds[layer][neuron] += learningRate * delta
			}
		}
	}
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return (1 - x) * x
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhDerivative(x float64) float64 {
	return 1 - x*x
}

func randoms(size int) []float64 {
	arr := make([]float64, size)
	for i := range arr {
		arr[i] = rand.Float64() - 0.5
	}
	return arr
}

func MeanSquaredError(errors []float64) float64 {
	sum := 0.0
	for _, err := range errors {
		sum += err * err
	}
	return sum / float64(len(errors))
}

func activationFunction(activation ActivationFunction) (func(float64) float64, func(float64) float64) {
	switch activation {
	case SIGMOID:
		return Sigmoid, SigmoidDerivative
	case TANH:
		return Tanh, TanhDerivative
	}
	return nil, nil
}
