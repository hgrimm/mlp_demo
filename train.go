package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
)

//--------------------------------------------------------
// Hilfsfunktionen zum Laden von MNIST
//--------------------------------------------------------

// loadMNIST lädt Bilder und Labels aus den IDX-Dateien.
// imageFile und labelFile sind die Pfade zu den entsprechenden MNIST-Dateien.
// Es liefert slices von Bildern (jede ein float64 slice der Länge 784) und Labels (one-hot Kodierung mit Länge 10).
func loadMNIST(imageFile, labelFile string) ([][]float64, [][]float64, error) {
	imgs, err := os.Open(imageFile)
	if err != nil {
		return nil, nil, err
	}
	defer imgs.Close()

	lbls, err := os.Open(labelFile)
	if err != nil {
		return nil, nil, err
	}
	defer lbls.Close()

	// Magic numbers und Header für Images
	var magic, numImages, rows, cols int32
	if err := binary.Read(imgs, binary.BigEndian, &magic); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(imgs, binary.BigEndian, &numImages); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(imgs, binary.BigEndian, &rows); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(imgs, binary.BigEndian, &cols); err != nil {
		return nil, nil, err
	}

	// Labels
	var magicLbl, numLabels int32
	if err := binary.Read(lbls, binary.BigEndian, &magicLbl); err != nil {
		return nil, nil, err
	}
	if err := binary.Read(lbls, binary.BigEndian, &numLabels); err != nil {
		return nil, nil, err
	}

	if numImages != numLabels {
		return nil, nil, fmt.Errorf("Anzahl Bilder und Labels stimmen nicht überein")
	}

	images := make([][]float64, numImages)
	labels := make([][]float64, numImages)

	for i := int32(0); i < numImages; i++ {
		img := make([]float64, rows*cols)
		for p := 0; p < int(rows*cols); p++ {
			var pixel uint8
			if err := binary.Read(imgs, binary.BigEndian, &pixel); err != nil {
				return nil, nil, err
			}
			// Normalisieren auf [0,1]
			img[p] = float64(pixel) / 255.0
		}

		var label uint8
		if err := binary.Read(lbls, binary.BigEndian, &label); err != nil {
			return nil, nil, err
		}

		// One-Hot-Encoding für Label
		onehot := make([]float64, 10)
		onehot[label] = 1.0

		images[i] = img
		labels[i] = onehot
	}

	return images, labels, nil
}

//--------------------------------------------------------
// Aktivierungsfunktionen und Hilfen
//--------------------------------------------------------

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Softmax für die Ausgabeschicht
func softmax(z []float64) []float64 {
	maxZ := math.Inf(-1)
	for _, val := range z {
		if val > maxZ {
			maxZ = val
		}
	}

	var sum float64
	expVals := make([]float64, len(z))
	for i, val := range z {
		ev := math.Exp(val - maxZ)
		expVals[i] = ev
		sum += ev
	}

	for i := range expVals {
		expVals[i] /= sum
	}
	return expVals
}

// Cross-Entropy-Loss
func crossEntropyLoss(yTrue, yPred []float64) float64 {
	// yTrue ist One-Hot, yPred Softmax.
	var loss float64
	for i := range yTrue {
		// Vermeide log(0) durch Hinzufügen einer kleinen Konstante.
		loss -= yTrue[i] * math.Log(yPred[i]+1e-12)
	}
	return loss
}

//--------------------------------------------------------
// MLP-Struktur
//--------------------------------------------------------

// Wir bauen ein MLP mit einer Hidden-Layer von z. B. 128 Neuronen
// Input: 784, Hidden: 128, Output: 10

type MLP struct {
	// Parameter
	W1 [][]float64
	b1 []float64
	W2 [][]float64
	b2 []float64
}

// initWeights initialisiert die Gewichte zufällig.
func initWeights(rows, cols int) [][]float64 {
	w := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		w[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			// Glorot-Initialisierung oder einfache Normalverteilung
			w[i][j] = rand.NormFloat64() * 0.01
		}
	}
	return w
}

func initBiases(dim int) []float64 {
	b := make([]float64, dim)
	return b
}

func NewMLP(inputDim, hiddenDim, outputDim int) *MLP {
	return &MLP{
		W1: initWeights(hiddenDim, inputDim),
		b1: initBiases(hiddenDim),
		W2: initWeights(outputDim, hiddenDim),
		b2: initBiases(outputDim),
	}
}

// Forward-Pass
func (m *MLP) Forward(x []float64) (z1, a1, z2, a2 []float64) {
	// Z1 = W1*x + b1
	hiddenDim := len(m.W1)
	inputDim := len(m.W1[0])
	z1 = make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		sum := 0.0
		for j := 0; j < inputDim; j++ {
			sum += m.W1[i][j] * x[j]
		}
		sum += m.b1[i]
		z1[i] = sum
	}

	// a1 = ReLU(z1)
	a1 = make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		a1[i] = relu(z1[i])
	}

	// Z2 = W2*a1 + b2
	outputDim := len(m.W2)
	z2 = make([]float64, outputDim)
	for i := 0; i < outputDim; i++ {
		sum := 0.0
		for j := 0; j < hiddenDim; j++ {
			sum += m.W2[i][j] * a1[j]
		}
		sum += m.b2[i]
		z2[i] = sum
	}

	a2 = softmax(z2)
	return
}

// Backpropagation
func (m *MLP) Backward(x, z1, a1, z2, a2, y []float64) (dW1 [][]float64, dB1 []float64, dW2 [][]float64, dB2 []float64) {
	// dLoss/dZ2 = (a2 - y)
	outputDim := len(a2)
	hiddenDim := len(a1)

	dZ2 := make([]float64, outputDim)
	for i := 0; i < outputDim; i++ {
		dZ2[i] = a2[i] - y[i]
	}

	// dW2 = dZ2 * a1^T
	dW2 = make([][]float64, outputDim)
	dB2 = make([]float64, outputDim)
	for i := 0; i < outputDim; i++ {
		dW2[i] = make([]float64, hiddenDim)
		for j := 0; j < hiddenDim; j++ {
			dW2[i][j] = dZ2[i] * a1[j]
		}
		dB2[i] = dZ2[i]
	}

	// dZ1 = W2^T * dZ2 * relu'(z1)
	dZ1 := make([]float64, hiddenDim)
	for j := 0; j < hiddenDim; j++ {
		sum := 0.0
		for i := 0; i < outputDim; i++ {
			sum += m.W2[i][j] * dZ2[i]
		}
		dZ1[j] = sum * reluDerivative(z1[j])
	}

	// dW1 = dZ1 * x^T
	inputDim := len(x)
	dW1 = make([][]float64, hiddenDim)
	dB1 = make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		dW1[i] = make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			dW1[i][j] = dZ1[i] * x[j]
		}
		dB1[i] = dZ1[i]
	}

	return
}

// Parameterupdate
func (m *MLP) Update(dW1 [][]float64, dB1 []float64, dW2 [][]float64, dB2 []float64, lr float64) {
	for i := range m.W1 {
		for j := range m.W1[i] {
			m.W1[i][j] -= lr * dW1[i][j]
		}
		m.b1[i] -= lr * dB1[i]
	}

	for i := range m.W2 {
		for j := range m.W2[i] {
			m.W2[i][j] -= lr * dW2[i][j]
		}
		m.b2[i] -= lr * dB2[i]
	}
}

// Predict gibt die vorhergesagte Klasse zurück
func (m *MLP) Predict(x []float64) int {
	_, _, _, a2 := m.Forward(x)
	maxVal := math.Inf(-1)
	maxIdx := 0
	for i, val := range a2 {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

// ComputeAccuracy berechnet die Genauigkeit auf einem Datensatz
func (m *MLP) ComputeAccuracy(X [][]float64, Y [][]float64) float64 {
	correct := 0
	for i, x := range X {
		pred := m.Predict(x)
		// Y[i] ist one-hot, pred sollte der Index sein, wo 1 ist
		trueLabel := 0
		for j, v := range Y[i] {
			if v == 1.0 {
				trueLabel = j
				break
			}
		}
		if pred == trueLabel {
			correct++
		}
	}
	return float64(correct) / float64(len(X))
}

// SaveModel speichert die Modellparameter (W1, b1, W2, b2) in eine JSON-Datei.
// filename: Pfad zur Zieldatei.
func SaveModel(filename string, W1 [][]float64, b1 []float64, W2 [][]float64, b2 []float64) error {
	modelData := map[string]interface{}{
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2,
	}

	jsonData, err := json.MarshalIndent(modelData, "", "  ")
	if err != nil {
		return fmt.Errorf("Fehler beim Serialisieren der Modellparameter: %v", err)
	}

	if err := ioutil.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("Fehler beim Schreiben der Datei: %v", err)
	}

	return nil
}

//--------------------------------------------------------
// Hauptprogramm
//--------------------------------------------------------

func main() {

	fmt.Println("Lade MNIST Trainingsdaten...")
	trainImages, trainLabels, err := loadMNIST("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatal("Fehler beim Laden der Trainingsdaten:", err)
	}

	fmt.Println("Lade MNIST Testdaten...")
	testImages, testLabels, err := loadMNIST("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte")
	if err != nil {
		log.Fatal("Fehler beim Laden der Testdaten:", err)
	}

	inputDim := 784 // 28*28
	hiddenDim := 512
	outputDim := 10
	learningRate := 0.09

	mlp := NewMLP(inputDim, hiddenDim, outputDim)

	epochs := 50
	batchSize := 50

	// print MLP hyperparameters
	fmt.Printf("MLP mit %d Eingabeneuronen, %d versteckten Neuronen und %d Ausgabeneuronen\n", inputDim, hiddenDim, outputDim)
	fmt.Printf("Lernrate: %.4f, %d Epochen, Batch-Größe: %d\n", learningRate, epochs, batchSize)

	for e := 0; e < epochs; e++ {
		// Shuffle der Trainingsdaten
		idxs := rand.Perm(len(trainImages))
		var totalLoss float64

		for i := 0; i < len(trainImages); i += batchSize {
			end := i + batchSize
			if end > len(trainImages) {
				end = len(trainImages)
			}

			// Mini-Batch
			dW1Sum := make([][]float64, hiddenDim)
			for h := 0; h < hiddenDim; h++ {
				dW1Sum[h] = make([]float64, inputDim)
			}
			dB1Sum := make([]float64, hiddenDim)

			dW2Sum := make([][]float64, outputDim)
			for o := 0; o < outputDim; o++ {
				dW2Sum[o] = make([]float64, hiddenDim)
			}
			dB2Sum := make([]float64, outputDim)

			batchCount := end - i
			var batchLoss float64

			for _, idx := range idxs[i:end] {
				x := trainImages[idx]
				y := trainLabels[idx]

				z1, a1, z2, a2 := mlp.Forward(x)
				l := crossEntropyLoss(y, a2)
				batchLoss += l

				dW1, dB1, dW2, dB2 := mlp.Backward(x, z1, a1, z2, a2, y)
				for hh := 0; hh < hiddenDim; hh++ {
					for jj := 0; jj < inputDim; jj++ {
						dW1Sum[hh][jj] += dW1[hh][jj]
					}
					dB1Sum[hh] += dB1[hh]
				}

				for oo := 0; oo < outputDim; oo++ {
					for hh := 0; hh < hiddenDim; hh++ {
						dW2Sum[oo][hh] += dW2[oo][hh]
					}
					dB2Sum[oo] += dB2[oo]
				}
			}

			// Durchschnittliche Gradienten des Mini-Batches
			for hh := 0; hh < hiddenDim; hh++ {
				for jj := 0; jj < inputDim; jj++ {
					dW1Sum[hh][jj] /= float64(batchCount)
				}
				dB1Sum[hh] /= float64(batchCount)
			}

			for oo := 0; oo < outputDim; oo++ {
				for hh := 0; hh < hiddenDim; hh++ {
					dW2Sum[oo][hh] /= float64(batchCount)
				}
				dB2Sum[oo] /= float64(batchCount)
			}

			// Parameterupdate
			mlp.Update(dW1Sum, dB1Sum, dW2Sum, dB2Sum, learningRate)
			totalLoss += batchLoss / float64(batchCount)
		}

		trainAcc := mlp.ComputeAccuracy(trainImages[:10000], trainLabels[:10000]) // aus Performancegründen nur einen Teil
		testAcc := mlp.ComputeAccuracy(testImages, testLabels)

		fmt.Printf("Epoche %d, Loss: %.4f, TrainAcc(10k): %.2f%%, TestAcc: %.2f%%\n",
			e, totalLoss/float64(len(trainImages)/batchSize), trainAcc*100, testAcc*100)
	}

	SaveModel("model.json", mlp.W1, mlp.b1, mlp.W2, mlp.b2)

}
