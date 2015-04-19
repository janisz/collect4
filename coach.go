package connect4

type Coach interface {
	Iteration()
	Error() float64
}

type BackPropagationCoach struct {
	perceptron *Perceptron
	input []Vector
	prediction []Vector
	deltas [][]float64
	errors [][]float64
	updates [][][]float64
	learningRate, momentum float64
}

func NewBackpropagationCoach(perceptron *Perceptron, input, prediction []Vector, learningRate, momentum float64) Coach {
	layersCount := len(perceptron.layers)
	deltas := make([][]float64, layersCount)
	errors := make([][]float64, layersCount)
	updates := make([][][]float64, layersCount)
	for i := range deltas {
		layerSize := perceptron.layers[i].Size()
		deltas[i] = make([]float64, layerSize)
		errors[i] = make([]float64, layerSize)
		updates[i] = make([][]float64, layerSize)
		for j := range updates[i] {
			updates[i][j] = make([]float64, perceptron.layers[i].neurons[0].weights.Length())
		}
	}
	return BackPropagationCoach{
		perceptron: perceptron,
		input: input,
		prediction: prediction,
		learningRate: learningRate,
		momentum: momentum,
		deltas: deltas,
		errors: errors,
		updates: updates,
	}
}
var bias bool
func (b BackPropagationCoach) Iteration() {
	for caseIndex, in := range b.input {
		log.Debug("Input: %s", in)
		prediction := b.prediction[caseIndex]
		outputs := b.perceptron.ComputeVerbose(in)

		log.Debug("Computing deltas...")
		layersCount := len(b.perceptron.layers)
		outputLayerIndex := layersCount - 1
		for layerIndex := outputLayerIndex; layerIndex >= 0; layerIndex-- {
			layer := b.perceptron.layers[layerIndex]
			layerOutput := outputs[layerIndex]
			for neuronIndex := range layer.neurons {
				neuronOutput := *layerOutput.At(neuronIndex)
				error := 0.0
				if (layerIndex == outputLayerIndex) {
					error = *prediction.At(neuronIndex) - neuronOutput
					log.Debug("Layer %d Neuron %d error %f ", layerIndex, neuronIndex, error)
				} else {
					nextLayerIndex := layerIndex + 1
					nextLayer := b.perceptron.layers[nextLayerIndex]
					for i, delta := range b.deltas[nextLayerIndex] {
						error += delta * *nextLayer.neurons[i].weights.At(neuronIndex)
					}
				}
				b.errors[layerIndex][neuronIndex] = error
				b.deltas[layerIndex][neuronIndex] = error * activationFunctionʼ(neuronOutput)
				log.Debug("Layer %d Neuron %d error %f ", layerIndex, neuronIndex, error)
			}
			log.Debug("Layer %d deltas %s", layerIndex, b.deltas[layerIndex])
		}

		log.Debug("Updating weights...")
		for layerIndex, layer := range b.perceptron.layers[1:layersCount] {
			layerInput := outputs[layerIndex]
			for neuronIndex, neuron := range layer.neurons {
				delta := b.deltas[layerIndex+1][neuronIndex]
				for i, inputElement := range layerInput.Raw() {
					update := b.updates[layerIndex+1][neuronIndex][i]
					log.Debug("Change = %f * %f * %f + %f * %f", b.learningRate, delta, inputElement, b.momentum, update)
					update = b.learningRate * delta * inputElement + b.momentum * update
					log.Debug("Change = %f", update)
					*(neuron.weights.At(i)) += update
					b.updates[layerIndex+1][neuronIndex][i] = update
				}
				if bias {
					neuron.bias += b.learningRate * delta
				}
			}
		}
	}
}


func (b BackPropagationCoach) Error() float64 {
	// mean squared error
	sum := 0.0;
	outputError := b.errors[len(b.errors)-1]
	for _, error := range outputError {
		sum += error * error
	}
	return (sum / float64(len(b.errors)));
}