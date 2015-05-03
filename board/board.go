package board
import (
	"fmt"
	"strings"
	"github.com/janisz/connect4/utils"
)

type Board struct {
	X       int
	Y       int
	Balance int
	Board   []float64
}

func NewBoard(x, y int) Board {
	b := Board{X: x, Y: y, Balance: 0}
	board := make([]float64, x*y)
	b.Board = board
	return b
}

func (b Board) SubBoard(x, y, width, length int) Board {
	subBoard := NewBoard(width, length)
	for j := 0 ; j < length ; j++ {
		for i := 0 ; i < width ; i++ {
			subBoard.Board[i+j*width] = b.Board[y*b.X + x + i + j * b.X]
		}
	}
	return subBoard
}

func (b Board) PrintBoard() {
	fmt.Printf("%s \n", strings.Join(utils.FloatsToStrings(b.Board[:], "%2.0f"), ","))
}

func (b Board) PrintHumanReadableBoard() {
	for i := 0; i < b.length(); i += b.Y {
		fmt.Printf("%s \n", strings.Join(utils.FloatsToStrings(b.Board[i:i+b.Y], "%2.0f"), ","))
	}
	fmt.Println("========")
}

func (b *Board) MakeMove(index int, symbol string) {
	b.Board[index] = 0
	if symbol == "1" {
		b.Board[index] = 1
		b.Balance += 1
	} else if symbol == "-1" {
		b.Board[index] = -1
		b.Balance -= 1
	}
}

func (b Board) IsMoveAllowed(fieldIndex int) bool {
	fieldBelowIndex := fieldIndex + b.X
	return fieldBelowIndex >= b.length() || b.Board[fieldBelowIndex] != 0
}

func (b Board) length() int {
	return len(b.Board)
}

