package board
import (
	"fmt"
	"strings"
)

type Board struct {
	X       int
	Y       int
	Balance int
	Board   []string
}

func NewBoard(x, y int) Board {
	b := Board{X: x, Y: y, Balance: 0}
	board := make([]string, x*y)
	for i := range board {
		board[i] = "0"
	}
	b.Board = board
	return b
}

func (b Board) PrintBoard() {
	fmt.Printf("%s \n", strings.Join(b.Board[:], ","))
}

func (b Board) PrintHumanReadableBoard() {
	for i := 0; i < b.X*b.Y; i += b.Y {
		fmt.Printf("%s \n", strings.Join(b.Board[i:i+b.Y], ","))
	}
	fmt.Println("========")
}

func (b *Board) MakeMove(index int, symbol string) {
	b.Board[index] = symbol
	if symbol == "1" {
		b.Balance += 1
	} else if symbol == "-1" {
		b.Balance -= 1
	}
}

func (b Board) IsMoveAllowed(fieldIndex int) bool {
	fieldBelowIndex := fieldIndex + b.X
	return fieldBelowIndex >= b.X*b.Y || b.Board[fieldBelowIndex] != "0"
}