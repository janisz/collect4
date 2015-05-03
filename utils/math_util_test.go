package utils

import "testing"

func Test_Round(t *testing.T) {
	if Round(123.555555, .5, 3) != 123.556 {
		t.Errorf("Something is wrong")
	}
}
