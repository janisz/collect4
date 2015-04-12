package collect4

import "errors"
import (
	"fmt"
	"math"
)

type Vector interface {
	Add(Vector) error
	Sum() float64
	Mul(float64)
	MulElements(Vector) error
	Length() int
	ElementAt(int) (float64, error)
	Apply(func(float64) float64)
	String() string
	Equals(x Vector) bool
	NearlyEquals(x Vector, eps float64) bool
	Copy() Vector
}

type SimpleVector struct {
	data []float64
}

func NewSimpleVector(data []float64) *SimpleVector {
	return &SimpleVector{data}
}

func NewZeroVector(size int) Vector {
	return &SimpleVector{make([]float64, size)}
}

func (v *SimpleVector) Copy() Vector {
	data := make([]float64, v.Length(), v.Length())
	copy(data, v.data)
	return NewSimpleVector(data)
}

func (v *SimpleVector) Equals(x Vector) bool {
	if v.Length() != x.Length() {
		return false
	}
	for i, lhs := range v.data {
		rhs, err := x.ElementAt(i)
		if err != nil {
			return false
		}
		if lhs != rhs {
			return false

		}
	}
	return true
}

func (v *SimpleVector) NearlyEquals(x Vector, eps float64) bool {
	if v.Length() != x.Length() {
		return false
	}
	for i, lhs := range v.data {
		rhs, err := x.ElementAt(i)
		if err != nil {
			return false
		}
		if !nearlyEqual(lhs, rhs, eps) {
			return false

		}
	}
	return true
}

func (v *SimpleVector) String() string {
	s := "[ "
	for _, element := range v.data {
		s += fmt.Sprintf("%f ", element)
	}
	return s + "]"
}
func (v *SimpleVector) Mul(x float64) {
	v.Apply(func(element float64) float64 {
		return element * x
	})
}

func (v *SimpleVector) Sum() float64 {
	sum := 0.0
	for _, element := range v.data {
		sum += element
	}
	return sum
}

func (v *SimpleVector) MulElements(x Vector) error {
	if v.Length() != x.Length() {
		return errors.New("Size mismatch")
	}
	for i, _ := range v.data {
		element, err := x.ElementAt(i)
		if err != nil {
			return errors.New("Size mismatch")
		}
		v.data[i] *= element
	}
	return nil
}

func (v *SimpleVector) Add(x Vector) error {
	if v.Length() != x.Length() {
		return errors.New("Size mismatch")
	}
	for i, _ := range v.data {
		element, err := x.ElementAt(i)
		if err != nil {
			return errors.New("Size mismatch")
		}
		v.data[i] += element
	}
	return nil
}

func (v *SimpleVector) Apply(transformation func(float64) float64) {
	for i, element := range v.data {
		v.data[i] = transformation(element)
	}
}

func (v *SimpleVector) ElementAt(position int) (float64, error) {
	if position < 0 || position >= v.Length() {
		return 0, errors.New("Argument out of bounds")
	}
	return v.data[position], nil
}

func (v *SimpleVector) Length() int {
	return len(v.data)
}

//http://stackoverflow.com/a/4915891/1387612
func nearlyEqual(a, b, epsilon float64) bool {
	absA := math.Abs(a)
	absB := math.Abs(b)
	diff := math.Abs(a - b)

	if a == b { // shortcut, handles infinities
		return true
	} else if a == 0 || b == 0 || diff < math.SmallestNonzeroFloat64 {
		// a or b is zero or both are extremely close to it
		// relative error is less meaningful here
		return diff < (epsilon * math.SmallestNonzeroFloat64)
	} else { // use relative error
		return diff/(absA+absB) < epsilon
	}
}
