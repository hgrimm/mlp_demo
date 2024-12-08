package main

import (
	"encoding/json"
	"fmt"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

// --------------------------------------------------------
// Laden des Modells (bereits definiert aus vorherigen Beispielen)
// --------------------------------------------------------

func LoadModel(filename string) (W1 [][]float64, b1 []float64, W2 [][]float64, b2 []float64, err error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("Fehler beim Lesen der Datei: %v", err)
	}

	var modelData map[string]interface{}
	if err := json.Unmarshal(data, &modelData); err != nil {
		return nil, nil, nil, nil, fmt.Errorf("Fehler beim Deserialisieren der JSON-Daten: %v", err)
	}

	to2DFloat := func(v interface{}) ([][]float64, error) {
		arr, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("Konvertierungsfehler: erwartet [][]float64")
		}
		result := make([][]float64, len(arr))
		for i := range arr {
			inner, ok := arr[i].([]interface{})
			if !ok {
				return nil, fmt.Errorf("Konvertierungsfehler: erwartet [][]float64")
			}
			result[i] = make([]float64, len(inner))
			for j := range inner {
				val, ok := inner[j].(float64)
				if !ok {
					return nil, fmt.Errorf("Konvertierungsfehler: erwartet float64")
				}
				result[i][j] = val
			}
		}
		return result, nil
	}

	to1DFloat := func(v interface{}) ([]float64, error) {
		arr, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("Konvertierungsfehler: erwartet []float64")
		}
		result := make([]float64, len(arr))
		for i, val := range arr {
			f, ok := val.(float64)
			if !ok {
				return nil, fmt.Errorf("Konvertierungsfehler: erwartet float64")
			}
			result[i] = f
		}
		return result, nil
	}

	// W1
	if w1Val, ok := modelData["W1"]; ok {
		W1, err = to2DFloat(w1Val)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	} else {
		return nil, nil, nil, nil, fmt.Errorf("W1 nicht gefunden")
	}

	// b1
	if b1Val, ok := modelData["b1"]; ok {
		b1, err = to1DFloat(b1Val)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	} else {
		return nil, nil, nil, nil, fmt.Errorf("b1 nicht gefunden")
	}

	// W2
	if w2Val, ok := modelData["W2"]; ok {
		W2, err = to2DFloat(w2Val)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	} else {
		return nil, nil, nil, nil, fmt.Errorf("W2 nicht gefunden")
	}

	// b2
	if b2Val, ok := modelData["b2"]; ok {
		b2Arr, err := to1DFloat(b2Val)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		if len(b2Arr) != 1 && len(b2Arr) != 10 {
			// In vorherigen Beispielen war b2 ein einzelner Wert,
			// falls wir jetzt 10 outputs haben, dann muss b2 ein Vektor sein.
			// Hier nehmen wir an, dass outputDim=10 (MNIST), dann muss b2 Länge 10 haben.
			if len(b2Arr) != 10 {
				return nil, nil, nil, nil, fmt.Errorf("b2 sollte Länge 10 haben für MNIST")
			}
			// Falls also das Netzwerk 10 Ausgabeneuronen hat, ist b2 ein Vektor.
			// In vorherigen Beispielen war b2 ein einzelner Wert, hier aber bei MNIST haben wir 10 Ausgaben:
			// also muss das Modell b2 als []float64 gespeichert haben.
			// Passen wir den Code an um damit umzugehen:
			if len(b2Arr) == 10 {
				// Erzeuge Fake-W2, b2 bereits in W2 und b2 kompatibel?
				// Hier vermuten wir, dass das Modell korrekt war.
				// Dann kann b2 als slice zurückgegeben werden. Wir müssen dann den Rückgabewert b2 als float64 ändern.
				// Passen wir den Function signature an? Sie war auf single float b2 ausgelegt.
				// In MNIST-Fall ist outputdim=10, also b2 ist []float64. Wir ändern den Rückgabetyp in der Signatur:
				// Signatur angepasst: b2 als float64 wird nun b2 []float64 zurückgeben.
				// Weiter unten wurde signiert. Bitte beachten Sie die Funktion oben wurde auch angepasst: b2 ist float64,
				// wir ändern die Signatur jetzt im Code und Annotations.

				// Da wir den Code bereits geschrieben haben, aktualisieren wir den Code so, dass b2 []float64 ist:
			}
		}
		// Da MNIST 10 Ausgaben hat, ist b2 ein Vektor:
		// Überschreiben wir einfach b2 als den ersten Wert:
		b2 = b2Arr
	} else {
		return nil, nil, nil, nil, fmt.Errorf("b2 nicht gefunden")
	}

	return W1, b1, W2, b2, nil
}

// --------------------------------------------------------
// Hilfsfunktionen für das Modell (Forward-Pass usw.)
// --------------------------------------------------------

// ReLU
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Softmax
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

// Forward Pass: Wir nehmen an, dass W2 und b2 nun ebenfalls zu 10 Dimensionen passen.
func forward(x []float64, W1 [][]float64, b1 []float64, W2 [][]float64, b2 []float64) []float64 {
	hiddenDim := len(W1)
	inputDim := len(W1[0])
	z1 := make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		sum := 0.0
		for j := 0; j < inputDim; j++ {
			sum += W1[i][j] * x[j]
		}
		sum += b1[i]
		z1[i] = sum
	}

	a1 := make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		a1[i] = relu(z1[i])
	}

	outputDim := len(W2)
	z2 := make([]float64, outputDim)
	for i := 0; i < outputDim; i++ {
		sum := 0.0
		for j := 0; j < hiddenDim; j++ {
			sum += W2[i][j] * a1[j]
		}
		sum += b2[i]
		z2[i] = sum
	}

	a2 := softmax(z2)
	return a2
}

// Predict gibt die vorhergesagte Klasse zurück
func predict(x []float64, W1 [][]float64, b1 []float64, W2 [][]float64, b2 []float64) int {
	a2 := forward(x, W1, b1, W2, b2)
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

// --------------------------------------------------------
// Bild laden und vorverarbeiten
// --------------------------------------------------------

// loadAndPreprocessImage lädt ein 28x28 PNG-Bild, wandelt es in ein Graustufen-Array um (0 bis 1 normalisiert)
func loadAndPreprocessImage(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("Fehler beim Öffnen des Bildes: %v", err)
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("Fehler beim Dekodieren des PNG: %v", err)
	}

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width != 28 || height != 28 {
		return nil, fmt.Errorf("Bildgröße muss 28x28 sein, ist aber %dx%d", width, height)
	}

	input := make([]float64, 28*28)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			c := img.At(x, y)
			// Falls c kein Grauwert ist, erstellen wir ihn aus dem Luminanzkanal
			r, g, b, _ := c.RGBA()
			// Convert to grayscale: einfacher Mittelwert oder Luminanz
			grayVal := 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8)
			// Normalisieren auf [0,1]
			grayValNorm := grayVal / 255.0
			input[y*28+x] = grayValNorm
		}
	}
	return input, nil
}

// --------------------------------------------------------
// Hauptfunktion: Inferenz-Client
// --------------------------------------------------------

func main() {
	rand.Seed(time.Now().UnixNano())

	// Beispiel: Wir laden das Modell "model.json"
	W1, b1, W2, b2, err := LoadModel("model.json")
	if err != nil {
		log.Fatalf("Fehler beim Laden des Modells: %v", err)
	}
	fmt.Println("Modell erfolgreich geladen.")

	// Laden eines einzelnen 28x28 PNG-Bildes, z. B. "digit.png"
	input, err := loadAndPreprocessImage("digit.png")
	if err != nil {
		log.Fatalf("Fehler beim Laden des Eingabebildes: %v", err)
	}
	fmt.Println("Eingabebild erfolgreich geladen und vorverarbeitet.")

	// Vorhersage treffen
	digit := predict(input, W1, b1, W2, b2)
	fmt.Printf("Das Modell erkennt die Ziffer als: %d\n", digit)
}
