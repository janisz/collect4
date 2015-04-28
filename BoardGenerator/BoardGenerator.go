package main

import (
	"fmt"
	"strings"
	"os"
	"strconv"
)

type Board struct {
	x int
	y int
	balance int
	board[] string
}

func NewBoard(x, y int) Board {
	b := Board{x: x, y: y, balance: 0}
	board := make([]string, x*y)
	for i := range board {
		board[i] = "0"
	}
	b.board = board
	return b
}

func generateAllPossibleBoards(x int, y int) {
	board := NewBoard(x, y)
	generateBoard(board, 0)
}

func generateBoard(board Board, depth int) {
	if depth == board.x*board.y {
//		only possible scenarios
		if (board.balance < 2 && board.balance > -2) {
			board.printBoard()
		}
		return
	}
	fieldIndex := board.x*board.y-1-depth;
	next_depth := depth + 1
	if (board.isMoveAllowed(fieldIndex)) {
		board.makeMove(fieldIndex, "-1")
		generateBoard(board, next_depth)
		board.makeMove(fieldIndex, "1")
		generateBoard(board, next_depth)
	}
	board.makeMove(fieldIndex, "0")
	generateBoard(board, next_depth)

}

func (b Board) printBoard() {
	fmt.Printf("%s \n", strings.Join(b.board[:], ","))
}

func (b Board) printHumanReadableBoard() {
	for i:=0; i<b.x*b.y; i+=b.y {
		fmt.Printf("%s \n", strings.Join(b.board[i:i+b.y], ","))
	}
	fmt.Println("========")
}

func (b *Board) makeMove(index int, symbol string) {
	b.board[index] = symbol
	if (symbol == "1") {
		b.balance += 1
	} else if (symbol == "-1") {
		b.balance -= 1
	}
}

func (b Board) isMoveAllowed(fieldIndex int) bool{
	fieldBelowIndex := fieldIndex + b.x
	return fieldBelowIndex >= b.x*b.y || b.board[fieldBelowIndex] != "0"
}

func main() {
	x := 4
	y := 4
	if (len(os.Args) == 3) {
		x, _ = strconv.Atoi(os.Args[1])
		y, _ = strconv.Atoi(os.Args[2])
	}
	fmt.Printf("%d %d", x, y)
	generateAllPossibleBoards(x, y)
}