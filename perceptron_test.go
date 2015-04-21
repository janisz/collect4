package connect4

import (
	"testing"
	"math"
)

func TestPerceptron(t *testing.T) {
	p := NewPerceptron([]int{4, 3, 2, 1}, false, sigmoid, sigmoidDerivative)
	p.Initialize()
	actual := p.Compute([]float64{-1, 0.5, 0, 1})
	if !compareFloat(0.547903, actual[0], 0.01) {
		t.Errorf("Expected %f but got %f", 0.0, actual)
	}
}

func TestPerceptronOnXOR(t *testing.T) {
	input := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 1},
		[]float64{0, 1},
	}

	ideal := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{0},
		[]float64{1},
	}

	p := NewPerceptron([]int{2, 3, 2, 1}, false, sigmoid, sigmoidDerivative)
	p.Initialize()
	error, iterationsWithoutBias := p.Learn(input, ideal, 0.7, 0.3, 1000, 0.0017)

	t.Logf("After %d iterfation got %f error", iterationsWithoutBias, error)

	for i, in := range input {
		actual := p.Compute(in)
		for j, actual_element := range actual {
			if !compareFloat(ideal[i][j], actual_element, 0.05) {
				t.Errorf("Expected %s but got %s", ideal[i], actual)
			}
		}
	}

	p = NewPerceptron([]int{2, 3, 2, 1}, true, sigmoid, sigmoidDerivative)
	p.Initialize()
	error, iterationsWithBias := p.Learn(input, ideal, 0.7, 0.3, 1000, 0.0017)

	t.Logf("After %d iterfation got %f error", iterationsWithBias, error)

	for i, in := range input {
		actual := p.Compute(in)
		for j, actual_element := range actual {
			if !compareFloat(ideal[i][j], actual_element, 0.05) {
				t.Errorf("Expected %s but got %s", ideal[i], actual)
			}
		}
	}

	if iterationsWithBias > iterationsWithoutBias {
		t.Error("Exected biased perceptron to learn faster")
	}
}

func BenchmarkLearn(b *testing.B) {
	p := NewPerceptron([]int{16, 12, 8, 4}, true, sigmoid, sigmoidDerivative)
	p.Initialize()
	b.ResetTimer()
	p.Learn([][]float64{randoms(16)}, [][]float64{randoms(4)}, 0.7, 0.3, b.N, 0);
}

func compareFloat(expected float64, actual float64, eps float64) bool {
	return math.Abs(expected - actual) < eps
}