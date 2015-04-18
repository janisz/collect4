package collect4
import "fmt"

type Neuron struct {
	weights Vector
}

func NewNeuron(input int) *Neuron {
	return &Neuron{
		weights: NewSimpleVector(make([]float64, input)),
	}
}

func (n *Neuron) Compute(signal Vector) float64 {
	output := signal.Copy()
	output.MulElements(n.weights)
	return output.Sum()
}

func (n *Neuron) String() string {
	return fmt.Sprintf("weights: %s", n.weights)
}
