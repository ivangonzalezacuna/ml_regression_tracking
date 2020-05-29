package main

import (
	ml "ml/logisticregression"

	log "github.com/sirupsen/logrus"
)

func init() {
	log.SetLevel(log.DebugLevel)
}

func main() {
	err := ml.LoadTrainData("./data/trackDataTrain.csv", "./data/trackDataTrain.csv")
	if err != nil {
		log.Errorf(err.Error())
	}

	err = ml.CreateBestModel()
	if err != nil {
		log.Errorf(err.Error())
	}

	predicData, err := ml.LoadPredictionDataFromCSV("./data/trackDataTest.csv")
	if err != nil {
		log.Errorf(err.Error())
	}

	prediction, err := ml.MakePrediction(predicData)
	if err != nil {
		log.Errorf(err.Error())
	}
	log.Infof("Final result: %v", prediction)
}
