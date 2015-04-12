package collect4

import "math"

type ActivationFunction func (float64) float64
type ActivationFunctionʼ func (float64) float64


var activationFunction ActivationFunction
var activationFunctionʼ ActivationFunction

func Sigmoid(x float64) float64 {
	return 1/(1 + math.Exp(-x))
}

func Sigmoidʼ(x float64) float64 {
	sigmoid := Sigmoid(x)
	return sigmoid*(1-sigmoid)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func Tanhʼ(x float64) float64 {
	tanh := Tanh(x)
	return 1 - tanh*tanh
}
