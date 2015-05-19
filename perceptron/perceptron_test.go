package perceptron

import (
	"encoding/json"
	"math"
	"testing"
)

func TestPerceptron(t *testing.T) {
	p := NewPerceptron([]int{4, 3, 2, 1}, false, SIGMOID)
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

	p := NewPerceptron([]int{2, 3, 2, 1}, false, SIGMOID)
	p.Initialize()
	error, iterationsWithoutBias := p.Learn(input, ideal, nil, nil, 0.7, 0.3, 1000, 0.0017)

	t.Logf("After %d iterfation got %f error", iterationsWithoutBias, error)

	for i, in := range input {
		actual := p.Compute(in)
		for j, actual_element := range actual {
			if !compareFloat(ideal[i][j], actual_element, 0.05) {
				t.Errorf("Expected %v but got %v", ideal[i], actual)
			}
		}
	}

	p = NewPerceptron([]int{2, 3, 2, 1}, true, SIGMOID)
	p.Initialize()
	error, iterationsWithBias := p.Learn(input, ideal, nil, nil, 0.7, 0.3, 1000, 0.0017)

	t.Logf("After %d iterfation got %f error", iterationsWithBias, error)

	for i, in := range input {
		actual := p.Compute(in)
		for j, actual_element := range actual {
			if !compareFloat(ideal[i][j], actual_element, 0.05) {
				t.Errorf("Expected %v but got %v", ideal[i], actual)
			}
		}
	}

	if iterationsWithBias > iterationsWithoutBias {
		t.Error("Exected biased perceptron to learn faster")
	}

	p = NewPerceptron([]int{2, 3, 2, 1}, true, TANH)
	p.Initialize()
	error, iterationsWithBiasAndTanh := p.Learn(input, ideal, nil, nil, 0.7, 0.3, 100, 0.001)

	t.Logf("After %d iterfation got %f error", iterationsWithBiasAndTanh, error)

	for i, in := range input {
		actual := p.Compute(in)
		for j, actual_element := range actual {
			if !compareFloat(ideal[i][j], actual_element, 0.05) {
				t.Errorf("Expected %v but got %v", ideal[i], actual)
			}
		}
	}

	if iterationsWithBiasAndTanh > iterationsWithBias {
		t.Error("Exected biased tanh perceptron to learn faster")
	}
}

func Test_ShouldStopVenErrorOnValidationRise(t *testing.T) {
	input := [][]float64{
		[]float64{-1},
		[]float64{-0.5},
		[]float64{0.00001}, //noise
		[]float64{0.5},
		[]float64{1},
	}

	ideal := [][]float64{
		[]float64{1},
		[]float64{0.25},
		[]float64{0.25}, //noise
		[]float64{0.25},
		[]float64{1},
	}

	validationInput := [][]float64{
		[]float64{0},
	}

	validationIdeal := [][]float64{
		[]float64{0},
	}

	p := NewPerceptron([]int{1, 2, 2, 1}, false, SIGMOID)
	p.Initialize()
	ee, iterationsWithoutValidation := p.Learn(input, ideal, nil, nil, 0.7, 0.3, 1000, 0.0)
	t.Log(ee)
	p = NewPerceptron([]int{1, 2, 2, 1}, false, SIGMOID)
	p.Initialize()
	ee, iterationsWithValidation := p.Learn(input, ideal, validationInput, validationIdeal, 0.7, 0.3, 1000, 0.0)
	t.Log(ee)
	if (iterationsWithoutValidation > iterationsWithValidation) {
		t.Errorf("There should be less iterations with validation %d >= %d", iterationsWithoutValidation, iterationsWithValidation)
	}

}

func Test_Json_Marshaling(t *testing.T) {
	original := NewPerceptron([]int{2, 3, 2, 1}, false, SIGMOID)
	original.Initialize()
	actual := &Perceptron{}
	encodedPerceptron, err := json.Marshal(original)
	t.Logf("%s", encodedPerceptron)
	json.Unmarshal(encodedPerceptron, actual)

	if err != nil || len(actual.Sizes) != len(original.Sizes) || len(actual.Weights) != len(original.Weights) {
		t.Errorf("Something went worng %s %s", err, encodedPerceptron)
	}

	actual.SetUp()

	if actual.Compute([]float64{0, 1})[0] != original.Compute([]float64{0, 1})[0] {
		t.Errorf("Something went worng %s %s", err, encodedPerceptron)
	}

}

func BenchmarkLearn(b *testing.B) {
	p := NewPerceptron([]int{16, 12, 8, 4}, true, SIGMOID)
	p.Initialize()
	b.ResetTimer()
	p.Learn([][]float64{randoms(16)}, [][]float64{randoms(4)}, nil, nil, 0.7, 0.3, b.N, 0)
}

func compareFloat(expected float64, actual float64, eps float64) bool {
	return math.Abs(expected-actual) < eps
}
