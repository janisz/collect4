package collect4

type Coach interface {
	Iteration()
	Error() float64
}

type BackPropagationCoach struct {
	perceptron *Perceptron
	input []Vector
	prediction []Vector
	deltas []Vector
	errors []Vector
	learnigRate float64
}

func NewBacpropagationCoach(perceptron *Perceptron, input, prediction []Vector, leariningRate float64) Coach {
	layersCount := len(perceptron.layers)
	deltas := make([]Vector, layersCount)
	errors := make([]Vector, layersCount)
	for i := range deltas {
		layerSize := perceptron.layers[i].Size()
		deltas[i] = NewZeroVector(layerSize)
		errors[i] = NewZeroVector(layerSize)
	}
	return BackPropagationCoach{
		perceptron: perceptron,
		input: input,
		prediction: prediction,
		learnigRate: leariningRate,
		deltas: deltas,
		errors: errors,
	}
}

func (b BackPropagationCoach) Iteration() {
	for caseIndex, in := range b.input {
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
					deltas := b.deltas[nextLayerIndex].Copy()
					for i, delta := range deltas.Raw() {
						error += delta * *nextLayer.neurons[i].weights.At(neuronIndex)
					}
				}
				*(b.errors[layerIndex].At(neuronIndex)) = error
				*(b.deltas[layerIndex].At(neuronIndex)) = error * activationFunctionʼ(neuronOutput)
				log.Debug("Layer %d Neuron %d error %f ", layerIndex, neuronIndex, error)
			}
			log.Debug("Layer %d deltas %s", layerIndex, b.deltas[layerIndex])
		}

		log.Debug("Updating weights...")
		for layerIndex, layer := range b.perceptron.layers[1:layersCount] {
			layerInput := outputs[layerIndex]
			for neuronIndex, neuron := range layer.neurons {
				delta := *(b.deltas[layerIndex+1].At(neuronIndex))
				for i, inputElement := range layerInput.Raw() {
					change := (b.learnigRate * delta * inputElement)
					log.Debug("Change = %f * %f * %f = %f", b.learnigRate, delta, inputElement, change)
					*(neuron.weights.At(i)) += change
				}
			}
		}
	}
}


func (b BackPropagationCoach) Error() float64 {
	ESS := NewZeroVector(b.prediction[0].Length());
	for i, in := range b.input {
		e := b.LocalError(in, b.prediction[i])
		e.Apply(func (x float64) float64 {
			return x*x
		})
		ESS.Add(e)
	}
	return ESS.Sum() / float64(ESS.Length())
}

func (b BackPropagationCoach) LocalError(in Vector, ideal Vector) Vector {
	output := b.perceptron.Compute(in)
	log.Debug("Perceptron Output for %s = %f", in, output)
	output.Mul(-1)
	output.Add(ideal)
	return output
}