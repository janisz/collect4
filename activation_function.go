package collect4

import "math"

//This type is function that will be applied to input in neuron
//First output is value of the function, and the second is it's derivative
//in givne point
type ActivationFunction interface {
	Value(float64) float64
	Derivative(float64) float64
}

var SIGMOID ActivationFunction = &Sigmoid{}

type Sigmoid struct{}

func (s *Sigmoid) Value(input float64) float64 {
	return math.Tanh(input)
}
func (s *Sigmoid) Derivative(input float64) float64 {
	return 1 - math.Tanh(input)*math.Tanh(input)
}
