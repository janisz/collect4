package collect4

import (
	"math/rand"
)

func (p *Perceptron) Learn(input, output []Vector) {
	for i, in := range input {
		prediction := output[i].Copy()
		actual := p.Compute(in)
		actual.Mul(-1.0)
		prediction.Add(actual)
		error := prediction.Raw()

		outputLayer := p.layers[len(p.layers)-1]
		for i := range outputLayer.neurons {
			neuron := &outputLayer.neurons[i]
			neuron.lastGradient = neuron.gradient
			neuron.gradient = activationFunctionʼ(neuron.output)*error[i]

			if (len(p.layers) > 2) {
				for j := range p.layers[len(p.layers)-2].neurons {
					previousNeuron := p.layers[len(p.layers)-2].neurons[j]
					*(neuron.weights.ElementAt(j)) +=  neuron.gradient * previousNeuron.output
				}
			} else {
				for j := range in.Raw() {
					(*neuron.weights.ElementAt(j)) +=  neuron.gradient * *in.ElementAt(j)
				}
			}

			neuron.weights.Apply(func (weight float64 ) float64 {
				return weight + neuron.gradient*neuron.output
			})
		}

		for i := len(p.layers)-2; i >= 0; i-- {
			layer := &p.layers[i]
			for j := range layer.neurons {
				neuron := &layer.neurons[j]
				neuron.lastGradient = neuron.gradient

				propagatedError := 0.0
				for _, nextNeuron := range p.layers[i+1].neurons {
					propagatedError += (*nextNeuron.weights.ElementAt(j))*nextNeuron.gradient
				}

				neuron.gradient = activationFunctionʼ(neuron.output)*propagatedError

				if (i > 1) {
					for k := range p.layers[i-1].neurons {
						previousNeuron := p.layers[i-1].neurons[k]
						(*neuron.weights.ElementAt(k)) +=  neuron.gradient * previousNeuron.output
					}
				} else {
					for k, inputElement := range in.Raw() {
						(*neuron.weights.ElementAt(k)) +=  neuron.gradient * inputElement
					}
				}
			}
		}
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


