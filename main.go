package main

import (
	"fmt"
	"github.com/janisz/connect4/board"
	"github.com/janisz/connect4/perceptron"
	"github.com/janisz/connect4/utils"
	"os"
)

func main() {
	argsWithoutProg := os.Args[1:]
	if len(argsWithoutProg) != 1 {
		fmt.Printf("Require resources file names")
		fmt.Printf("Ex: 4x4.csv 4x7.csv 6x7.csv")
		return
	}

	/* TODO: NOT FINISHED YET
	 *	Sanitize data (add normalisation)
	 *	Finish generating perceptrons (think about classification approach)
	 *	Combine everything in one huge network
	 *	Add basic UI
	 */

	fourOnFour := utils.ReadCsvToFloats(argsWithoutProg[0])
	block := perceptron.NewPerceptron([]int{4 * 4, 12, 8, 1}, true, perceptron.TANH)
	block.Initialize()
	block.Learn(fourOnFour[:16], fourOnFour[16:], 0.6, 0.1, 1000, 0.001)
	utils.Save(block, "block.json")

	fourOnSevenRaw := utils.ReadCsvToFloats(argsWithoutProg[1])
	fourOnSeven := make([][]float64, len(fourOnSevenRaw))
	for index, board_4x7 := range fourOnSevenRaw {
		fourOnSeven[index] = make([]float64, 4+1)
		for i := 0; i < 4; i++ {
			fourOnSeven[i] = block.Compute(board_4x7[i*16 : (i+1)*16])
		}
		fourOnSeven[index][4] = board_4x7[4*16]
	}

	column := perceptron.NewPerceptron([]int{4, 16, 8, 1}, true, perceptron.TANH)
	column.Initialize()
	column.Learn(fourOnSeven[:4], fourOnSeven[4:], 0.6, 0.1, 1000, 0.001)
	utils.Save(column, "column.json")

	sixOnSevenRaw := utils.ReadCsvToFloats(argsWithoutProg[2])
	sixOnSeven := make([][]float64, len(sixOnSevenRaw))
	for index, board_6x7 := range sixOnSevenRaw {
		sixOnSeven[index] = make([]float64, 3+1)
		for i := 0; i < 3; i++ {
			sixOnSeven[i] = block.Compute(board_6x7[i*16 : (i+1)*16])
		}
		sixOnSeven[index][3] = board_6x7[3*16]
	}

	decider := perceptron.NewPerceptron([]int{3, 16, 8, 1}, true, perceptron.TANH)
	decider.Initialize()
	decider.Learn(sixOnSeven[:3], sixOnSeven[3:], 0.6, 0.1, 1000, 0.001)
	utils.Save(decider, "decider.json")
}

func play(board board.Board) int {

	//TODO: Load files once
	block := &perceptron.Perceptron{}
	utils.Load(block, "block.json")
	column := &perceptron.Perceptron{}
	utils.Load(column, "column.json")
	decider := &perceptron.Perceptron{}
	utils.Load(decider, "decider.json")

	columnInput := make([][]float64, 3)
	for j := 0; j < 4; j++ {
		columnInput[j] = make([]float64, 4)
		for i := 0; i < 4; i++ {
			columnInput[j][i] = block.Compute(board.SubBoard(i, j, 4, 4).Board)[0]
		}
	}

	deciderInput := make([]float64, 3)
	for i := 0; i < 3; i++ {
		deciderInput[i] = block.Compute(columnInput[i])[0]
	}

	return int(decider.Compute(deciderInput)[0])
}
