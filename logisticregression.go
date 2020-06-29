package logisticregression

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
)

type (
	confusionMatrix struct {
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

	//TrainData has all the train and test data
	TrainData struct {
		xTrain [][]float64
		yTrain []float64
		xTest  [][]float64
		yTest  []float64
	}

	// ModelData has all the info related to the trained model
	ModelData struct {
		Model             *linear.Logistic
		DecissionBoundary float64
		Iterations        int
		Accuracy          float64
		ConfusionMatrix   confusionMatrix
	}
)

// LoadTrainDataFromCSV loads the train & test data from CSV files
func LoadTrainDataFromCSV(trainFilePath, testFilePath string) (TrainData, error) {
	log.Infof("Loading Train & Test data from CSV files...")
	var err error
	xTrain, yTrain, err := base.LoadDataFromCSV(trainFilePath)
	if err != nil {
		return TrainData{}, err
	}
	xTest, yTest, err := base.LoadDataFromCSV(testFilePath)
	if err != nil {
		return TrainData{}, err
	}

	return TrainData{xTrain, yTrain, xTest, yTest}, nil
}

// LoadTrainDataRaw loads the train & test data directly from 2D float64 arrays
func LoadTrainDataRaw(trainData, testData [][]float64) (TrainData, error) {
	log.Infof("Loading Train & Test data from 2D float64 array...")
	if len(trainData) == 0 || len(testData) == 0 {
		return TrainData{}, fmt.Errorf("Received empty dataset")
	}

	trainSize := len(trainData[0])
	for _, v := range trainData {
		if trainSize != len(v) {
			return TrainData{}, fmt.Errorf("Train dataset size mismatch")
		}
	}

	testSize := len(testData[0])
	for _, v := range testData {
		if testSize != len(v) {
			return TrainData{}, fmt.Errorf("Test dataset size mismatch")
		}
	}

	if trainSize != testSize {
		return TrainData{}, fmt.Errorf("Train & Test datasets have different sizes")
	}

	var xTrain, xTest [][]float64
	var yTrain, yTest []float64

	for _, v := range trainData {
		xTrain = append(xTrain, v[:trainSize-1])
		yTrain = append(yTrain, v[trainSize-1])
	}

	for _, v := range testData {
		xTest = append(xTest, v[:testSize-1])
		yTest = append(yTest, v[testSize-1])
	}

	return TrainData{xTrain, yTrain, xTest, yTest}, nil
}

// MakePrediction makes a prediction using a trained model and an input data
func (m *ModelData) MakePrediction(xPrediction [][]float64) ([]int, error) {
	if m.Model == nil {
		return nil, fmt.Errorf("Can't make a prediction based on nil Model")
	}

	log.Infof("Making new prediction based on collected data...")
	var finalPrediction []int
	trainDataSize := len(m.Model.Theta()) - 1
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
		prediction, err := m.Model.Predict(xPrediction[i])
		if err != nil {
			return nil, err
		}
		if prediction[0] >= m.DecissionBoundary {
			finalPrediction = append(finalPrediction, 1)
		} else {
			finalPrediction = append(finalPrediction, 0)
		}
	}
	return finalPrediction, nil
}

func readConfig() {
	userDir, err := osUser.Current()
	if err != nil {
		log.Errorf(err.Error())
	}

	configDir := path.Join(userDir.HomeDir, ".config", "ml-system")
	_, err = os.Stat(configDir)
	if os.IsNotExist(err) {
		errDir := os.MkdirAll(configDir, 0755)
		if errDir != nil {
			log.Errorf(err.Error())
		}
	}

	cfgFile := path.Join(configDir, "config.toml")
	viper.SetConfigFile(cfgFile)
	if err := viper.ReadInConfig(); err != nil {
		log.Errorf("[Init] Unable to read config from file %s: %s", cfgFile, err.Error())
	} else {
		log.Infof("[Init] Read configuration from file %s", cfgFile)
	}
}

// CreateBestModel executes a loop in order to find the most accurate model
func (t *TrainData) CreateBestModel() (ModelData, error) {
	log.Infof("Looking for the best Logistic Regression Model...")
	readConfig()
	viper.SetDefault("ml.iterations", -1)
	iter := viper.GetInt("ml.iterations")
	viper.SetDefault("ml.decissionBoundary", -1)
	db := viper.GetInt("ml.decissionBoundary")

	var maxAccuracyModel *linear.Logistic
	var maxAccuracyDb float64
	var maxAccuracyIter int
	var maxAccuracy float64
	var maxAccuracyCM confusionMatrix

	if iterations != -1 && db != -1 {
		cm, model, err := findBestModel(0.0001, 0.0, iter, db, t.xTrain, t.xTest, t.yTrain, t.yTest)
		if err != nil {
			return ModelData{}, err
		}
		return ModelData{model, db, iter, cm.accuracy, cm}, nil
	}

	//Try different parameters to get the best model
	for iter = 100; iter < 3300; iter += 500 {
		for db = 0.05; db < 1.0; db += 0.01 {
			cm, model, err := findBestModel(0.0001, 0.0, iter, db, t.xTrain, t.xTest, t.yTrain, t.yTest)
			if err != nil {
				return ModelData{}, err
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

	viper.Set("ml.iterations", iter)
	viper.Set("ml.decissionBoundary", db)
	viper.WriteConfig()

	m := ModelData{maxAccuracyModel, maxAccuracyDb, maxAccuracyIter, maxAccuracy, maxAccuracyCM}

	return m, nil
}

func findBestModel(learningRate float64, regularization float64, iterations int, decisionBoundary float64, xTrain, xTest [][]float64, yTrain, yTest []float64) (confusionMatrix, *linear.Logistic, error) {
	cm := confusionMatrix{}
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
		return confusionMatrix{}, nil, err
	}

	// Evaluate the Model on the Test data
	for i := range xTest {
		prediction, err := model.Predict(xTest[i])
		if err != nil {
			return confusionMatrix{}, nil, err
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
	return cm, model, nil
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
