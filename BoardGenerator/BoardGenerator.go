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
	if depth == 8 {
		printBoard(vector)
		return
	}
	next_depth := depth + 1
	fieldIndex := 15-depth;
	fieldBelowIndex := fieldIndex + 4;
	if (fieldBelowIndex > 15 || vector[fieldBelowIndex] != "0") {
		vector[fieldIndex] = "-1"
		generateBoard(vector, next_depth)
		vector[fieldIndex] = "1"
		generateBoard(vector, next_depth)
	}
	vector[fieldIndex] = "0"
	generateBoard(vector, next_depth)

}

func printBoard(vector [16]string) {
	for i:=0; i<16; i+=4 {
		fmt.Printf("%s \n", strings.Join(vector[i:i+4], ","))
	}
	fmt.Println("========")
}

func main() {
	generateAllPossibleBoards()
}