package main

import (
	"github.com/janisz/connect4/board"
	"os"
	"strconv"
	"math/rand"
)

func generateAllPossibleBoards(x int, y int) {
	board := board.NewBoard(x, y)
	generateBoard(board, 0)
}

func generateBoard(board board.Board, depth int) {
	p := rand.Float32()
	if (depth == board.X*board.Y) || (p < 0.25) {
		//		only possible scenarios
		if board.Balance < 2 && board.Balance > -2 {
			board.PrintBoard()
		}
		return
	}
	fieldIndex := board.X*board.Y - 1 - depth
	next_depth := depth + 1
	if board.IsMoveAllowed(fieldIndex) {
		board.MakeMove(fieldIndex, "-1")
		generateBoard(board, next_depth)
		board.MakeMove(fieldIndex, "1")
		generateBoard(board, next_depth)
	}
	board.MakeMove(fieldIndex, "0")
	generateBoard(board, next_depth)

}

func main() {
	x := 4
	y := 4
	if len(os.Args) == 3 {
		x, _ = strconv.Atoi(os.Args[1])
		y, _ = strconv.Atoi(os.Args[2])
	}
	generateAllPossibleBoards(x, y)
}
