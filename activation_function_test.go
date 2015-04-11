package collect4

import "testing"
import "math"

func TestSigmoidFunction(t *testing.T) {
	input := []float64{-1, 0, 1}
	for _, in := range input {
		expectedValue := math.Tanh(in)
		expectedDerivative := 1 - math.Tanh(in)*math.Tanh(in)
		actualValue, actualDerivative := Sigmoid(in)
		if expectedValue != actualValue() || expectedDerivative != actualDerivative() {
			t.Errorf("Expected %f, %f but got %f, %f",
				expectedValue, expectedDerivative, actualValue(), actualDerivative())
		}

	}
}
