package collect4

import "math"

//This type is function that will be applied to input in neuron
//First output is value of the function, and the second is it's derivative
//in givne point
type ActivationFunction func(float64) (func() float64, func() float64)

func Sigmoid(input float64) (func() float64, func() float64) {
	return func() float64 {
			return math.Tanh(input)
		}, func() float64 {
			return 1 - math.Tanh(input)*math.Tanh(input)
		}
}
