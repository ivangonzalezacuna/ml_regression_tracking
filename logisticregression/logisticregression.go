package logisticregression

import (
	"fmt"
	"io/ioutil"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"

	log "github.com/sirupsen/logrus"
)

// Max accuracy data for a traned model
var maxAccuracyModel *linear.Logistic
var maxAccuracyDb float64
var maxAccuracyIter int
var maxAccuracy float64
var maxAccuracyCM *ConfusionMatrix

//Train & Test data
var xTrain [][]float64
var yTrain []float64
var xTest [][]float64
var yTest []float64

// ConfusionMatrix describes a confusion matrix
type ConfusionMatrix struct {
	positive      int
	negative      int
	truePositive  int
	trueNegative  int
	falsePositive int
	falseNegative int
	recall        float64
	precision     float64
	accuracy      float64
}

func (cm ConfusionMatrix) String() string {
	return fmt.Sprintf("\tPositives: %d\n\tNegatives: %d\n\tTrue Positives: %d\n\tTrue Negatives: %d\n\tFalse Positives: %d\n\tFalse Negatives: %d\n\n\tRecall: %.2f\n\tPrecision: %.2f\n\tAccuracy: %.2f\n",
		cm.positive, cm.negative, cm.truePositive, cm.trueNegative, cm.falsePositive, cm.falseNegative, cm.recall, cm.precision, cm.accuracy)
}

// LoadTrainDataFromCSV loads the CSV files for training and testing to find the best Model
func LoadTrainDataFromCSV(trainFilePath, testFilePath string) error {
	log.Infof("Loading Train & Test data from CSV files...")
	var err error
	xTrain, yTrain, xTest, yTest = nil, nil, nil, nil
	xTrain, yTrain, err = base.LoadDataFromCSV(trainFilePath)
	if err != nil {
		return err
	}
	xTest, yTest, err = base.LoadDataFromCSV(testFilePath)
	if err != nil {
		return err
	}
	return nil
}

// LoadTrainDataRaw loads the CSV files for training and testing to find the best Model
func LoadTrainDataRaw(trainData, testData [][]float64) error {
	log.Infof("Loading Train & Test data from 2D float64 array...")
	if len(trainData) == 0 || len(testData) == 0 {
		return fmt.Errorf("Received empty dataset")
	}

	trainSize := len(trainData[0])
	for _, v := range trainData {
		if trainSize != len(v) {
			return fmt.Errorf("Train dataset size mismatch")
		}
	}

	testSize := len(testData[0])
	for _, v := range testData {
		if testSize != len(v) {
			return fmt.Errorf("Test dataset size mismatch")
		}
	}

	if trainSize != testSize {
		return fmt.Errorf("Train & Test datasets have different sizes")
	}

	xTrain, yTrain, xTest, yTest = nil, nil, nil, nil

	for _, v := range trainData {
		xTrain = append(xTrain, v[:trainSize-1])
		yTrain = append(yTrain, v[trainSize-1])
	}

	for _, v := range testData {
		xTest = append(xTest, v[:testSize-1])
		yTest = append(yTest, v[testSize-1])
	}

	return nil
}

// LoadPredictionDataFromCSV loads the CSV file in order to make a prediction. Maybe we don't even need this
func LoadPredictionDataFromCSV(predictionPath string) ([][]float64, error) {
	log.Infof("Loading Predicition data from CSV file...")
	var err error
	xPrediction, _, err := base.LoadDataFromCSV(predictionPath)
	if err != nil {
		return nil, err
	}
	return xPrediction, nil
}

// MakePrediction makes a prediction of a list of data
func MakePrediction(xPrediction [][]float64) ([]int, error) {
	log.Infof("Making new prediction based on collected data...")
	var finalPrediction []int
	trainDataSize := len(maxAccuracyModel.Theta()) - 1

	if len(xPrediction) == 0 {
		return nil, fmt.Errorf("Empty prediction dataset")
	}

	log.Debugf("Model size: %v", trainDataSize)
	log.Debugf("Prediction data size: %v", len(xPrediction[0]))

	for _, v := range xPrediction {
		if len(v) != trainDataSize {
			return nil, fmt.Errorf("Prediction dataset has different size than Train dataset")
		}
	}

	for i := range xPrediction {
		if trainDataSize != len(xPrediction[i]) {
			return nil, fmt.Errorf("Trained data and prediction data size mismatch")
		}
		prediction, err := maxAccuracyModel.Predict(xPrediction[i])
		if err != nil {
			return nil, err
		}
		if prediction[0] >= maxAccuracyDb {
			finalPrediction = append(finalPrediction, 1)
		} else {
			finalPrediction = append(finalPrediction, 0)
		}
	}
	return finalPrediction, nil
}

// CreateBestModel executes a loop in order to find the most accurate model
func CreateBestModel() error {
	log.Infof("Searching for the best Logistic Regression Model...")

	//Try different parameters to get the best model
	for iter := 100; iter < 3300; iter += 500 {
		for db := 0.05; db < 1.0; db += 0.01 {
			cm, model, err := findBestModel(0.0001, 0.0, iter, db, xTrain, xTest, yTrain, yTest)
			if err != nil {
				return err
			}
			if cm.accuracy > maxAccuracy {
				maxAccuracy = cm.accuracy
				maxAccuracyCM = cm
				maxAccuracyDb = db
				maxAccuracyModel = model
				maxAccuracyIter = iter
			}
		}
	}

	log.Debugf("Maximum accuracy: %.2f", maxAccuracy)
	// log.Debugf("with Model: %s\n\n", maxAccuracyModel)
	// log.Debugf("with Confusion Matrix:\n%s\n\n", maxAccuracyCM)
	log.Debugf("with Decision Boundary: %.2f", maxAccuracyDb)
	log.Debugf("with Num Iterations: %d", maxAccuracyIter)

	// if err := plotData(xTrain, yTrain); err != nil {
	// 	return err
	// }

	return nil
}

// func plotData(xTest [][]float64, yTest []float64) error {
// 	p, err := plot.New()
// 	if err != nil {
// 		return err
// 	}
// 	p.Title.Text = "Tracking Results"
// 	p.X.Label.Text = "X"
// 	p.Y.Label.Text = "Y"
// 	p.X.Max = 120
// 	p.Y.Max = 120

// 	positives := make(plotter.XYs, len(yTest))
// 	negatives := make(plotter.XYs, len(yTest))
// 	for i := range xTest {
// 		if yTest[i] == 1.0 {
// 			positives[i].X = xTest[i][0]
// 			positives[i].Y = xTest[i][1]
// 		}
// 		if yTest[i] == 0.0 {
// 			negatives[i].X = xTest[i][0]
// 			negatives[i].Y = xTest[i][1]
// 		}
// 	}

// 	err = plotutil.AddScatters(p, "Negatives", negatives, "Positives", positives)
// 	if err != nil {
// 		return err
// 	}
// 	if err := p.Save(10*vg.Inch, 10*vg.Inch, "result.png"); err != nil {
// 		return err
// 	}
// 	return nil
// }

func findBestModel(learningRate float64, regularization float64, iterations int, decisionBoundary float64, xTrain, xTest [][]float64, yTrain, yTest []float64) (*ConfusionMatrix, *linear.Logistic, error) {
	cm := ConfusionMatrix{}
	for _, y := range yTest {
		if y == 1.0 {
			cm.positive++
		}
		if y == 0.0 {
			cm.negative++
		}
	}

	// Instantiate and Learn the Model
	model := linear.NewLogistic(base.BatchGA, learningRate, regularization, iterations, xTrain, yTrain)
	model.Output = ioutil.Discard
	err := model.Learn()
	if err != nil {
		return nil, nil, err
	}

	// Evaluate the Model on the Test data
	for i := range xTest {
		prediction, err := model.Predict(xTest[i])
		if err != nil {
			return nil, nil, err
		}
		y := int(yTest[i])
		positive := prediction[0] >= decisionBoundary

		if y == 1 && positive {
			cm.truePositive++
		}
		if y == 1 && !positive {
			cm.falseNegative++
		}
		if y == 0 && positive {
			cm.falsePositive++
		}
		if y == 0 && !positive {
			cm.trueNegative++
		}
	}

	// Calculate Evaluation Metrics
	cm.recall = float64(cm.truePositive) / float64(cm.positive)
	cm.precision = float64(cm.truePositive) / (float64(cm.truePositive) + float64(cm.falsePositive))
	cm.accuracy = float64(float64(cm.truePositive)+float64(cm.trueNegative)) / float64(float64(cm.positive)+float64(cm.negative))
	return &cm, model, nil
}
