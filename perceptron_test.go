package connect4

import (
	"testing"
	"fmt"
	"math"
)

func TestPerceptron(t *testing.T) {
	ConfigureDebugLogging()
	p := NewPerceptron(4, 1, []int{3, 2})
	activationFunction = Tanh
	actual := p.Compute(NewSimpleVector([]float64{-1, 0.5, 0, 1}))
	expected := NewZeroVector(1)
	if !actual.Equals(expected) {
		t.Errorf("Expected %s but got %s for %s", expected, actual, p)
	}
}

func TestPerceptronComuteWithoutHiddenLayers(t *testing.T) {
	ConfigureDebugLogging()
	p := NewPerceptron(2, 1, []int{})
	activationFunction = Tanh
	p.layers[0].neurons[0].weights = NewSimpleVector([]float64{1, -0.04930600})
	actual := p.Compute(NewSimpleVector([]float64{0.5, -1}))
	expected := NewSimpleVector([]float64{0.5})
	if !actual.NearlyEquals(expected, 1E-6) {
		t.Errorf("Expected %s but got %s for %s", expected, actual, p)
	}
}

func TestPerceptronComuteLayers(t *testing.T) {
	ConfigureDebugLogging()
	p := NewPerceptron(2, 1, []int{1})
	activationFunction = Tanh
	p.layers[0].neurons[0].weights = NewSimpleVector([]float64{1, -0.04930600})
	p.layers[1].neurons[0].weights = NewSimpleVector([]float64{-0.200671})
	actual := p.Compute(NewSimpleVector([]float64{0.5, -1}))
	expected := NewSimpleVector([]float64{-0.1})
	if !actual.NearlyEquals(expected, 1E-6) {
		t.Errorf("Expected %s but got %s for %s", expected, actual, p)
	}
}

func TestPerceptronInitialization(t *testing.T) {
	ConfigureDebugLogging()
	p := NewPerceptron(4, 1, []int{3, 2})
	activationFunction = Tanh
	p.Initialize(64)
	actual := p.Compute(NewSimpleVector([]float64{-1, 0.5, 0, 1}))
	expected := NewZeroVector(1)
	if actual.Equals(expected) {
		t.Errorf("NOT Expected %s but got %s for %s", expected, actual, p)
	}
}

func TestPerceptronBackpropagation(t *testing.T) {
	ConfigureDebugLogging()
	//given:
	activationFunction = Sigmoid
	activationFunctionʼ = Sigmoidʼ
	bias = false
	p := NewPerceptron(2, 1, []int{2})
	p.layers[0].neurons[0].weights = NewSimpleVector([]float64{0.1, 0.8})
	p.layers[0].neurons[1].weights = NewSimpleVector([]float64{0.4, 0.6})
	p.layers[1].neurons[0].weights = NewSimpleVector([]float64{0.3, 0.9})

	input := NewSimpleVector([]float64{0.35, 0.9})
	expected := NewSimpleVector([]float64{0.5})

	coach := NewBackpropagationCoach(p, []Vector{input}, []Vector{expected}, 1, 0)

	actual := p.Compute(input)
	if actual.NearlyEquals(expected, 0.182051) {
		t.Errorf("Expected %s got %s for %s", expected, actual, p)
	}

	//when:
	p.Learn(coach, 0, 1)
	log.Debug("Perceptron: %s", p)

	//then:
	actual = p.Compute(input)
	if !actual.NearlyEquals(expected, 0.182051) || actual.NearlyEquals(expected, 0.18204) {
		t.Errorf("Expected %s but got %s for %s", expected, actual, p)
	}
}

func TestPerceptronBackpropagation2(t *testing.T) {
	ConfigureDebugLogging()
	//given:
	activationFunction = Sigmoid
	activationFunctionʼ = Sigmoidʼ
	bias = true
	p := NewPerceptron(2, 2, []int{3})

	p.layers[0].neurons[0].bias = 0.1
	p.layers[0].neurons[1].bias = 0.2
	p.layers[0].neurons[2].bias = 0.5
	p.layers[1].neurons[0].bias = -0.1
	p.layers[1].neurons[1].bias = 0.6

	log.Debug("p %s", p)
	p.layers[0].neurons[0].weights = NewSimpleVector([]float64{0.1, -0.2})
	p.layers[0].neurons[1].weights = NewSimpleVector([]float64{0.0, 0.2})
	p.layers[0].neurons[2].weights = NewSimpleVector([]float64{0.3, -0.4})
	p.layers[1].neurons[0].weights = NewSimpleVector([]float64{-0.4, 0.1, 0.6})
	p.layers[1].neurons[1].weights = NewSimpleVector([]float64{0.2, -0.1, -0.2})

	input := NewSimpleVector([]float64{0.6, 0.1})
	expected := NewSimpleVector([]float64{0.53, 0.62})
	ideal := NewSimpleVector([]float64{1, 0.0})

	coach := NewBackpropagationCoach(p, []Vector{input}, []Vector{ideal}, 0.1, 0)

	actual := p.Compute(input)
	if actual.NearlyEquals(expected, 0.005) {
		t.Errorf("Expected %s got %s for %s", expected, actual, p)
	}

	//when:
	p.Learn(coach, 0, 1)
	log.Debug("Perceptron: %s", p)

	w36expected := -0.394
	w36actual := *p.layers[1].neurons[0].weights.At(0)
	w37expected := 0.192
	w37actual := *p.layers[1].neurons[1].weights.At(0)
	log.Debug("weight 36 %f, %f", w36expected, w36actual)
	if !compareFloat(w36expected, w36actual, 0.005) {
		t.Errorf("Expected %f but got %f for %s", w36expected, w36actual, p)
	}
	log.Debug("weight 37 %f, %f", w37expected, w37actual)
	if !compareFloat(w37expected, w37actual, 0.005) {
		t.Errorf("Expected %f but got %f for %s", w37expected, w37actual, p)
	}

	w13expected := 0.0989
	w13actual := *p.layers[0].neurons[0].weights.At(0)
	log.Debug("weight 13 %f, %f", w13expected, w13actual)
	if !compareFloat(w13expected, w13actual, 0.00005) {
		t.Errorf("Expected %f but got %f for %s", w13expected, w13actual, p)
	}
}

func TestPerceptronBackpropagationXOR(t *testing.T) {
	ConfigureInfoLogging()
	//given:
	activationFunction = Sigmoid
	activationFunctionʼ = Sigmoidʼ
	bias = true
	p := NewPerceptron(2, 1, []int{3,2})
	p.Initialize(127)

	input := []Vector{
		NewSimpleVector([]float64{0, 0}),
		NewSimpleVector([]float64{0, 1}),
		NewSimpleVector([]float64{1, 1}),
		NewSimpleVector([]float64{1, 0}),
	}
	expected := []Vector{
		NewSimpleVector([]float64{0}),
		NewSimpleVector([]float64{1}),
		NewSimpleVector([]float64{0}),
		NewSimpleVector([]float64{1}),
	}

	coach := NewBackpropagationCoach(p, input, expected, 0.7, 0.3)

	//when:
	p.Learn(coach, 0.001, 5000)
	log.Info("Perceptron %s", p)
	//then:
	if coach.Error() > 0.01 {
		t.Errorf("Error is too big: %f", coach.Error())
	}

	resultsSummary := "\nINPUT\t|\tACTUAL\t|\tEXPECTED\n"
	for i, in := range input {
		resultsSummary += fmt.Sprintf("%s\t|\t%s\t|\t%s\n", in, p.Compute(in), expected[i])
	}
	fmt.Println(resultsSummary)
}

func TestPerceptronBackpropagationAND(t *testing.T) {
	ConfigureInfoLogging()
	//given:
	activationFunction = Sigmoid
	activationFunctionʼ = Sigmoidʼ
	bias = false
	p := NewPerceptron(2, 1, []int{3,2})
	p.Initialize(127)

	input := []Vector{
		NewSimpleVector([]float64{0, 0}),
		NewSimpleVector([]float64{0, 1}),
		NewSimpleVector([]float64{1, 1}),
		NewSimpleVector([]float64{1, 0}),
	}
	expected := []Vector{
		NewSimpleVector([]float64{0}),
		NewSimpleVector([]float64{0}),
		NewSimpleVector([]float64{1}),
		NewSimpleVector([]float64{0}),
	}

	coach := NewBackpropagationCoach(p, input, expected, 0.9, 0.9)

	//when:
	p.Learn(coach, 0.001, 50000)
	//then:
	log.Info("Perceptron %s", p)
	if !p.Compute(input[2]).NearlyEquals(expected[2], 0.3) {
		t.Errorf("Expected %s but got %s", expected[2], p.Compute(input[2]))
	}

	resultsSummary := "\nINPUT\t|\tACTUAL\t|\tEXPECTED\n"
	for i, in := range input {
		resultsSummary += fmt.Sprintf("%s\t|\t%s\t|\t%s\n", in, p.Compute(in), expected[i])
	}
	log.Info(resultsSummary)
}


func compareFloat(expected float64, actual float64, eps float64) bool{
	return math.Abs(expected - actual) < eps
}
