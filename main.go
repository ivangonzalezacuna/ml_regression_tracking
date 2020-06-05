package main

import (
	ml "ml/logisticregression"

	log "github.com/sirupsen/logrus"
)

func init() {
	log.SetLevel(log.DebugLevel)
}

func main() {
	trainData, err := ml.LoadTrainDataFromCSV("./data/trackDataTrain.csv", "./data/trackDataTrain.csv")
	if err != nil {
		log.Errorf(err.Error())
	}

	trainModel, err := trainData.CreateBestModel()
	if err != nil {
		log.Errorf(err.Error())
	}

	log.Debugf("ModelData: %#v", trainModel)

	predicData, err := ml.LoadPredictionDataFromCSV("./data/trackDataTrain.csv")
	if err != nil {
		log.Errorf(err.Error())
	}

	prediction, err := trainModel.MakePrediction(predicData)
	if err != nil {
		log.Errorf(err.Error())
	}
	log.Infof("Final result: %v", prediction)
}
