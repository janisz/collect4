package board

import "testing"

func Test_sub_board_happy_path(t *testing.T) {
	b := NewBoard(4, 4)
	b.Board = []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}

	subBoard := b.SubBoard(1, 1, 2, 2)
	expected := []float64{
		6, 7,
		10, 11,
	}

	for i, actual := range subBoard.Board {
		if expected[i] != actual {
			t.Errorf("Expected %s but got %s", expected, subBoard.Board)
		}
	}

}
