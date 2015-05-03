package utils
import (
	"github.com/janisz/connect4/perceptron"
	"encoding/json"
	"io/ioutil"
)

func Save(perceptron perceptron.Perceptron, filename string ) {
	json, err := json.MarshalIndent(perceptron, "", "\t")
	if err != nil {
		panic(err)
	}
	err = ioutil.WriteFile(filename, json, 0644)
	if err != nil {
		panic(err)
	}
}
