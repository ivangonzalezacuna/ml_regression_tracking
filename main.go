package main

import (
	"fmt"
	"os"
	"path/filepath"

	logReg "ml/logisticregression"

	log "github.com/sirupsen/logrus"
)

var dirPath string

func init() {
	log.SetLevel(log.DebugLevel)
	var err error
	dirPath, err = filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Errorf(err.Error())
	}
	dirPath = fmt.Sprintf("%v/model.txt", dirPath)
}

func main() {
	logReg.InitLogisticRegression()
	err := logReg.LoadTrainData("./data/trackDataTrain.csv", "./data/trackDataTrain.csv")
	if err != nil {
		log.Errorf(err.Error())
	}

	err = logReg.CreateBestModel(dirPath)
	if err != nil {
		log.Errorf(err.Error())
	}

	err = logReg.LoadPredictionData("./data/trackDataTrain.csv")
	if err != nil {
		log.Errorf(err.Error())
	}
	prediction, err := logReg.MakePrediction()
	if err != nil {
		log.Errorf(err.Error())
	}
	log.Infof("Final result: %v", prediction)
}
