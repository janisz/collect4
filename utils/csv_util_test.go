package utils
import "testing"

func Test_StringsToInt_Happy_Path(t *testing.T) {
	ints := stringsToFloats([]string{"0","1","2","3"})
	for i, element := range ints {
		if (float64(i) != element) {
			t.Errorf("Expected %d but got %d", i, element)
		}
	}
}

func Test_StringsToInt_For_Empty_Slice(t *testing.T) {
	ints := stringsToFloats([]string{})
	if (len(ints) != 0) {
		t.Errorf("Expected zero lengt but got %s", ints)
	}
}

func Test_StringsToInt_For_Non_Ints(t *testing.T) {
	ints := stringsToFloats([]string{"1", "qwerty"})
	if (len(ints) != 0) {
		t.Errorf("Expected zero lengt but got %s", ints)
	}
}