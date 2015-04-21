package connect4
import (
	"math"
	"math/rand"
)

type Perceptron struct {
	bias bool
	sizes []int
	thresholds [][]float64
	outputs [][]float64
	errors [][]float64
	deltas [][]float64
	weights [][][]float64
	changes [][][]float64
	activationFunction func(float64) float64
	activationFunctionDerivative func(float64) float64
}

func NewPerceptron(sizes []int, bias bool, activationFunction, activationFunctionDerivative func(float64) float64) Perceptron {
	return Perceptron{
		bias: bias,
		sizes: sizes,
		activationFunction: activationFunction,
		activationFunctionDerivative: activationFunctionDerivative,
	}
}

func (p *Perceptron) Initialize() {

	rand.Seed(1024)

	p.weights = make([][][]float64, len(p.sizes))
	p.changes = make([][][]float64, len(p.sizes))

	p.thresholds =  make([][]float64, len(p.sizes))
	p.outputs =  make([][]float64, len(p.sizes))
	p.errors =  make([][]float64, len(p.sizes))
	p.deltas = make([][]float64, len(p.sizes))

	for i := range p.weights {

		p.thresholds[i] = make([]float64, p.sizes[i])
		p.outputs[i] = make([]float64, p.sizes[i])
		p.errors[i] = make([]float64, p.sizes[i])
		p.deltas[i] = make([]float64, p.sizes[i])

		p.weights[i] = make([][]float64, p.sizes[i])
		p.changes[i] = make([][]float64, p.sizes[i])

		for j := 0; j < p.sizes[i]; j++ {
			if (i == 0) {
				p.weights[i][j] = randoms(0)
				p.changes[i][j] = make([]float64, 0)
			} else {
				p.weights[i][j] = randoms(p.sizes[i-1])
				p.changes[i][j] = make([]float64, p.sizes[i-1])
			}

		}
	}
}

func (p *Perceptron) outputLayer() int {
	return len(p.sizes) - 1
}


func (p *Perceptron) Compute(input []float64) []float64 {
	p.outputs[0] = input
	for layer := 1; layer <= p.outputLayer(); layer++ {
		for neuron := 0; neuron < p.sizes[layer]; neuron++ {
			neuronWeights := p.weights[layer][neuron];
			sum := 0.0;
			if p.bias {
				sum += p.thresholds[layer][neuron]
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
		for neuron := 0; neuron < p.sizes[layer]; neuron++ {
			output := p.outputs[layer][neuron]
			error := 0.0
			if (layer == p.outputLayer()) {
				error = ideal[neuron] - output
			} else {
				deltas := p.deltas[layer + 1]
				for i, delta := range deltas {
					error += delta * p.weights[layer + 1][i][neuron]
				}
			}
			p.errors[layer][neuron] = error
			p.deltas[layer][neuron] = error * p.activationFunctionDerivative(output)
		}
	}
}

func (p *Perceptron) updateWeights(learningRate float64, momentum float64) {
	for layer := 1; layer <= p.outputLayer(); layer++ {
		incoming := p.outputs[layer - 1]

		for neuron := 0; neuron < p.sizes[layer]; neuron++ {
			delta := p.deltas[layer][neuron]

			for previousLayerNeuron := range incoming {
				change := p.changes[layer][neuron][previousLayerNeuron]

				change = learningRate * delta * incoming[previousLayerNeuron] + momentum * change

				p.changes[layer][neuron][previousLayerNeuron] = change
				p.weights[layer][neuron][previousLayerNeuron] += change
			}
			if p.bias {
				p.thresholds[layer][neuron] += learningRate * delta
			}
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x));
}

func sigmoidDerivative(x float64) float64 {
	return (1 - x) * x;
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