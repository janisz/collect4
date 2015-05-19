package main

import (
	"fmt"
	"github.com/janisz/connect4/board"
	"github.com/janisz/connect4/perceptron"
	"github.com/janisz/connect4/utils"
	"os"
	"github.com/op/go-logging"
)

var log = logging.MustGetLogger("example")

var format = logging.MustStringFormatter(
"%{color}%{time:15:04:05.000} %{shortfunc} ▶ %{level:.4s} %{id:03x}%{color:reset} %{message}",
)

func main() {

	backend := logging.NewLogBackend(os.Stderr, "", 0)
	backendFormatter := logging.NewBackendFormatter(backend, format)
	backendLeveled := logging.AddModuleLevel(backendFormatter)
	backendLeveled.SetLevel(logging.DEBUG, "")
	logging.SetBackend(backendLeveled)

	args := os.Args[1:]

	if len(args) == 0 {
		fmt.Println(
		"1. Build network: go run main.go 4x4.csv 4x7.csv 6x7.csv\n",
		"2. Play: go run main.go board.csv",
		)
	}

	if len(args) == 3 {
		build(args[0], args[1], args[2])
		return
	} else if len(args) == 1 {
		sixOnSeven := utils.ReadCsvToFloats(args[0])
		for i, line := range sixOnSeven {
			board_6x7 := board.NewBoard(6, 7)
			board_6x7.Board = line[:6*7]
			fmt.Printf("Game %d\n", i)
			board_6x7.PrintHumanReadableBoard()
			fmt.Printf("Move %d expected %d\n", play(board_6x7), denormalize(line[6*7]))
		}

	}

}

func build(fourOnFourFilename, fourOnSevenFilename, sixOnSevenFilename string) {

	fourOnFour := utils.ReadCsvToFloats(fourOnFourFilename)

	inputs_4x4 := make([][]float64, len(fourOnFour))
	outputs_4x4 := make([][]float64, len(fourOnFour))
	for i, line := range fourOnFour {
		inputs_4x4[i] = make([]float64, 4*4)
		outputs_4x4[i] = make([]float64, 1)
		inputs_4x4[i] = line[:16]
		outputs_4x4[i][0] = line[16]
	}

	block := perceptron.NewPerceptron([]int{4 * 4, 12, 8, 1}, true, perceptron.TANH)
	block.Initialize()
	error, iterations := block.Learn(inputs_4x4, outputs_4x4, nil, nil, 0.6, 0.1, 1000, 0.001)
	log.Info("Learning block endend with error %f after %d iterations", error, iterations)
	utils.Save(block, "block.json")

	fourOnSeven := utils.ReadCsvToFloats(fourOnSevenFilename)
	inputs_4x7, outputs_4x7 := generateFourOnSevenBoard(block, fourOnSeven)

	column := perceptron.NewPerceptron([]int{4, 16, 8, 1}, true, perceptron.TANH)
	column.Initialize()
	error, iterations = column.Learn(inputs_4x7, outputs_4x7, nil, nil, 0.6, 0.1, 1000, 0.001)
	log.Info("Learning column endend with error %f after %d iterations", error, iterations)
	utils.Save(column, "column.json")

	sixOnSeven := utils.ReadCsvToFloats(sixOnSevenFilename)
	inputs_6x7, outputs_6x7 := generateSixOnSevenBoard(column, sixOnSeven)

	decider := perceptron.NewPerceptron([]int{3, 16, 8, 1}, true, perceptron.TANH)
	decider.Initialize()
	error, iterations = decider.Learn(inputs_6x7, outputs_6x7, nil, nil, 0.01, 0.05, 10000, 0.001)
	log.Info("Learning decider endend with error %f after %d iterations", error, iterations)
	utils.Save(decider, "decider.json")
}

func generateFourOnSevenBoard(block perceptron.Perceptron, fourOnSeven [][]float64) (inputs_4x7, outputs_4x7 [][]float64) {
	inputs_4x7 = make([][]float64, len(fourOnSeven))
	outputs_4x7 = make([][]float64, len(fourOnSeven))
	for i, line := range fourOnSeven {
		inputs_4x7[i] = make([]float64, 4)
		outputs_4x7[i] = make([]float64, 1)
		board_4x7 := board.NewBoard(4, 7)
		board_4x7.Board = line[:4*7]
		log.Debug("Board%s", board_4x7)
		for j := 0; j < 4; j++ {
			subBoard := board_4x7.SubBoard(0, j, 4, 4)
			log.Debug("Block board %d%s", j, subBoard)
			inputs_4x7[i][j] = block.Compute(subBoard.Board)[0]
		}
		outputs_4x7[i][0] = line[4*7]
	}
	return inputs_4x7, outputs_4x7
}

func generateSixOnSevenBoard(column perceptron.Perceptron, sixOnSeven [][]float64) (inputs_6x7, outputs_6x7 [][]float64) {
	inputs_6x7 = make([][]float64, len(sixOnSeven))
	outputs_6x7 = make([][]float64, len(sixOnSeven))
	for i, line := range sixOnSeven {
		inputs_6x7[i] = make([]float64, 3)
		outputs_6x7[i] = make([]float64, 1)
		board_6x7 := board.NewBoard(6, 7)
		board_6x7.Board = line[:6*7]
		log.Debug("Board%s", board_6x7)
		for j := 0; j < 3; j++ {
			subBoard := board_6x7.SubBoard(j, 0, 4, 7)
			log.Debug("Column board %d%s", j, subBoard)
			inputs_6x7[i][j] = column.Compute(subBoard.Board)[0]
		}
		outputs_6x7[i][0] = line[6*7]
	}
	return inputs_6x7, outputs_6x7
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

	for j := 0; j < 3; j++ {
		columnInput[j] = make([]float64, 4)
		for i := 0; i < 4; i++ {
			subBoard := board.SubBoard(j, i, 4, 4)
			log.Debug("SubBoard %d/%d%s", j, i, subBoard)
			columnInput[j][i] = block.Compute(subBoard.Board)[0]
		}
		log.Info("Column %d input %s", j, utils.FloatsToStrings(columnInput[j], "%2.0f"))
	}

	deciderInput := make([]float64, 3)
	for i := 0; i < 3; i++ {
		deciderInput[i] = column.Compute(columnInput[i])[0]
	}
	log.Info("Decider input input %s", utils.FloatsToStrings(deciderInput, "%2.0f"))

	deciderOutput := decider.Compute(deciderInput)[0]
	return denormalize(deciderOutput)
}

func denormalize(deciderOutput float64) int {
	return int(utils.Round(2*deciderOutput+2, 0.5, 0))
}
