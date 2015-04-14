package collect4

type Coach interface {
	Iteration()
	Error() float64
}

type BackPropagationCoach struct {
	perceptron *Perceptron
	input []Vector
	prediction []Vector
	learnigRate float64
}

func NewBacpropagationCoach(perceptron *Perceptron, input, prediction []Vector, leariningRate float64) Coach {
	return BackPropagationCoach{
		perceptron: perceptron,
		input: input,
		prediction: prediction,
		learnigRate: leariningRate,
	}
}

func (b BackPropagationCoach) Iteration() {
	log.Info("Iteration step")
	p := b.perceptron
	log.Debug("Perceptron: %s", p)
	layersCount := len(p.layers)
	learningRate := b.learnigRate
	for i, in := range b.input {
		error := b.LocalError(in, b.prediction[i]).Raw()
		outputLayer := p.layers[layersCount-1]
		log.Debug("Local Error: %s", error)
		for i, neuron := range outputLayer.neurons {
			log.Debug("Updating output layer weights")
			neuron.lastGradient = neuron.gradient
			neuron.gradient = activationFunctionʼ(neuron.output)*error[i]
			log.Debug("Neuron gradient: %f * %f", activationFunctionʼ(neuron.output), error[i])
			log.Debug("Neuron gradient for output %f %f", activationFunction(neuron.output), neuron.gradient)

			if (layersCount > 1) {
				for j := range p.layers[layersCount-2].neurons {
					previousNeuron := p.layers[layersCount-2].neurons[j]
					*(neuron.weights.At(j)) += learningRate * neuron.gradient * previousNeuron.output
				}
			} else { //No hidden layer
				for j := range in.Raw() {
					(*neuron.weights.At(j)) +=  learningRate * neuron.gradient * *in.At(j)
				}
			}
			log.Debug("Perceptron: %s", p)
		}

		for i := layersCount-2; i >= 0; i-- {
			layer := p.layers[i] //TODO: Change it to for range
			log.Debug("Updating %d hidden layers weights", i)
			for j, neuron := range layer.neurons {
				neuron.lastGradient = neuron.gradient

				propagatedError := 0.0
				for _, nextNeuron := range p.layers[i+1].neurons {
					propagatedError += (*nextNeuron.weights.At(j))*nextNeuron.gradient
					log.Debug("Propagated error += %f x %f", (*nextNeuron.weights.At(j)), nextNeuron.gradient)
				}
				log.Debug("Propagated error %f", propagatedError)
				neuron.gradient = activationFunctionʼ(neuron.output)*propagatedError
				log.Debug("Neuron gradient for output %f %f", activationFunction(neuron.output), neuron.gradient)

				if (i > 1) {
					for k := range p.layers[i-1].neurons {
						previousNeuron := p.layers[i-1].neurons[k]
						(*neuron.weights.At(k)) += learningRate * neuron.gradient * previousNeuron.output
					}
				} else {
					for j := range in.Raw() {
						(*neuron.weights.At(j)) +=  learningRate * neuron.gradient * *in.At(j)
					}
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