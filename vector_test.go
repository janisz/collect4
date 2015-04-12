package collect4

import "testing"

func TestReturnNewVector(t *testing.T) {
	var v Vector = NewSimpleVector([]float64{})
	if v == nil {
		t.Error("This should never happen")
	}
}

func TestSaveVectorAsString(t *testing.T) {
	var v Vector = NewSimpleVector([]float64{7, 8, 9})
	if "[ 7.000000 8.000000 9.000000 ]" != v.String() {
		t.Error("Expected [ 7.000000 8.000000 9.000000 ] but got %s", v.String())
	}
}
func TestMulVectorsElements(t *testing.T) {
	a := NewSimpleVector([]float64{7, 8, 9})
	b := NewSimpleVector([]float64{1, 2, 3})
	expected := NewSimpleVector([]float64{7, 16, 27})
	err := a.MulElements(b)
	actual := a
	if err != nil {
		t.Errorf("Got error: %s", err)
	}
	if !actual.Equals(expected) {
		t.Errorf("Expected %s but got %s", expected, actual)
	}
}
func TestMulVector(t *testing.T) {
	var expected Vector = NewSimpleVector([]float64{21, 24, 27})
	var actual Vector = NewSimpleVector([]float64{7, 8, 9})

	actual.Mul(3)

	if !actual.Equals(expected) {
		t.Errorf("Expected %s but got %s", expected, actual)
	}
}
func TestSumVector(t *testing.T) {
	var v Vector = NewSimpleVector([]float64{7, 8, 9})
	if v.Sum() != 24 {
		t.Errorf("Expected 21 but got %f", v.Sum())
	}
}
func TestVectorNotEquals(t *testing.T) {
	cases := []struct {
		a, b []float64
	}{
		{[]float64{1, 2}, []float64{1}},
		{[]float64{1, 2}, []float64{1, 3}},
		{[]float64{1, }, []float64{1, 2}},
	}
	for _, c := range cases {
		a := NewSimpleVector(c.a)
		b := NewSimpleVector(c.b)
		if a.Equals(b) {
			t.Errorf("%s %s are NOT same", a, b)
		}
	}
}

func TestVectorEquals(t *testing.T) {
	cases := []struct {
		a, b []float64
	}{
		{[]float64{1, 2}, []float64{1, 2}},
		{[]float64{}, []float64{}},
	}
	for _, c := range cases {
		a := NewSimpleVector(c.a)
		b := NewSimpleVector(c.b)
		if !a.Equals(b) {
			t.Errorf("%s %s are same", a, b)
		}
	}
}

func TestVectorNotNearlyEquals(t *testing.T) {
	cases := []struct {
		a, b []float64
	}{
		{[]float64{1, 2}, []float64{1}},
		{[]float64{1, 2}, []float64{1, 2.01}},
		{[]float64{1, }, []float64{1, 2}},
	}
	for _, c := range cases {
		a := NewSimpleVector(c.a)
		b := NewSimpleVector(c.b)
		if a.NearlyEquals(b, 0.001) {
			t.Errorf("%s %s are NOT same", a, b)
		}
	}
}

func TestVectorNearlyEquals(t *testing.T) {
	cases := []struct {
		a, b []float64
	}{
		{[]float64{0.99, 2}, []float64{1, 2.01}},
		{[]float64{}, []float64{}},
	}
	for _, c := range cases {
		a := NewSimpleVector(c.a)
		b := NewSimpleVector(c.b)
		if !a.NearlyEquals(b, 0.011) {
			t.Errorf("%s %s are same", a, b)
		}
	}
}
