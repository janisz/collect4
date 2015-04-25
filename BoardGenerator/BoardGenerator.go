package main

import (
	"fmt"
	"strings"
)

func generateAllPossibleBoards() {
	board := [16]string{}
	for i := range board {
		board[i] = "0"
	}
	generateBoard(board, 0)
}

func generateBoard(vector [16]string, depth int) {
	if depth == 2 {
		fmt.Printf("%s\n", strings.Join(vector[:], ","))
		return
	}
	next_depth := depth + 1
	vector[depth] = "-1"
	generateBoard(vector, next_depth)
	vector[depth] = "1"
	generateBoard(vector, next_depth)
	vector[depth] = "0"
	generateBoard(vector, next_depth)

}

func main() {
	generateAllPossibleBoards()
}