package main

import (
	"fmt"
	"strings"
)

type Board struct {
	x int
	y int
	board[] string
}

func NewBoard(x, y int) Board {
	b := Board{x: x, y: y}
	board := make([]string, x*y)
	for i := range board {
		board[i] = "0"
	}
	b.board = board
	return b
}

func generateAllPossibleBoards() {
	board := NewBoard(4, 4)
	generateBoard(board, 0)
}

func generateBoard(board Board, depth int) {
	if depth == 8 {
		board.printBoard()
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
	for i:=0; i<b.x*b.y; i+=b.y {
		fmt.Printf("%s \n", strings.Join(b.board[i:i+b.y], ","))
	}
	fmt.Println("========")
}

func (b *Board) makeMove(index int, symbol string) {
	b.board[index] = symbol
}

func (b Board) isMoveAllowed(fieldIndex int) bool{
	fieldBelowIndex := fieldIndex + b.x
	return fieldBelowIndex >= b.x*b.y || b.board[fieldBelowIndex] != "0"
}

func main() {
	generateAllPossibleBoards()
}