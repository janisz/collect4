package main

import (
	"fmt"
	"strings"
)

func generate_all_possible_boards() {
	board := [16]string{}
	for i := range board {
		board[i] = "0"
	}
	generate_board(board, 0)
}

func generate_board(vector [16]string, depth int) {
	if depth == 2 {
		fmt.Printf("%s\n", strings.Join(vector[:], ","))
		return
	}
	next_depth := depth + 1
	vector[depth] = "-1"
	generate_board(vector, next_depth)
	vector[depth] = "1"
	generate_board(vector, next_depth)
	vector[depth] = "0"
	generate_board(vector, next_depth)

}

func main() {
	generate_all_possible_boards()
}