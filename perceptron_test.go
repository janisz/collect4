package collect4

import (
	"testing"
)

func TestPerceptron(t *testing.T) {
	p := NewPerceptron(4, 1, []int{3,2})
	actual := p.Compute(NewSimpleVector([]float64{-1, 0.5, 0, 1}))
	expected := NewZeroVector(1)
	if !actual.Equals(expected) {
		t.Errorf("Expected %s but got %s for %s", expected, actual, p)
	}
}

func TestPerceptronComuteWithoutHiddenLayers(t *testing.T) {
	p := NewPerceptron(2, 1, []int{})
	p.layers[0].neurons[0].weights = NewSimpleVector([]float64{1, -0.04930600})
	actual := p.Compute(NewSimpleVector([]float64{0.5, -1}))
	expected := NewSimpleVector([]float64{0.5})
	if !actual.NearlyEquals(expected, 1E-6) {
		t.Errorf("Expected %s but got %s for %s", expected, actual, p)
	}
}

