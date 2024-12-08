package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

var (
	useImageMagick bool
	listenAddr     string
	W1             [][]float64
	b1             []float64
	W2             [][]float64
	b2             []float64
)

func init() {
	flag.BoolVar(&useImageMagick, "useimagemagick", true, "Use ImageMagick for image conversion")
	flag.StringVar(&listenAddr, "listen", ":7766", "Address to listen on")
}

func main() {
	flag.Parse()

	// Modell nur einmal laden
	var err error
	W1, b1, W2, b2, err = LoadModel("model.json")
	if err != nil {
		log.Fatalf("Fehler beim Laden des Modells: %v", err)
	}
	log.Println("Modell erfolgreich geladen.")

	http.HandleFunc("/", serveHTML)
	http.HandleFunc("/upload", handleUpload)
	// start the server an log if it fails
	log.Printf("Server listening on %s", listenAddr)
	if err := http.ListenAndServe(listenAddr, nil); err != nil {
		log.Fatalf("Fehler beim Starten des Servers: %v", err)
	}
}

func serveHTML(w http.ResponseWriter, r *http.Request) {
	// This HTML uses a 28x28 canvas, but scaled up by a factor of 8 (224x224) via CSS.
	// The background is set to black and the pen is white.
	html := `<!DOCTYPE html>
<html>
<head>
<title>Canvas 28x28</title>
<style>
  body { font-family: sans-serif; }
  #canvasContainer {
    position: relative;
    width: 224px;
    height: 224px;
    border: 1px solid #000;
  }
  canvas {
    image-rendering: pixelated;
    width: 224px;
    height: 224px;
    cursor: crosshair;
  }
</style>
</head>
<body>
<h1>Draw on the 28x28 Canvas</h1>
<div id="canvasContainer">
<canvas id="drawingCanvas" width="28" height="28"></canvas>
</div>
<br>
<button id="submitBtn">Submit Image</button>
<button id="clearBtn">Clear Canvas</button>
<p id="antwortAbschnitt"></p>

<script>
  const canvas = document.getElementById('drawingCanvas');
  const ctx = canvas.getContext('2d');
  const PEN_SIZE = 2; // Pen size 2x2 pixels
  
  // Function to clear the canvas (fill it with black)
  function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // Initially clear the canvas to black
  clearCanvas();

  let drawing = false;
  let currentColor = '#FFFFFF'; // default pen color is white for left click

  // Prevent the context menu on right click
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
  });

  canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    if (e.button === 0) {
      // Left click draws white
      currentColor = '#FFFFFF';
    } else if (e.button === 2) {
      // Right click draws black
      currentColor = '#000000';
    }
    drawPixel(e);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (drawing) drawPixel(e);
  });

  canvas.addEventListener('mouseup', () => {
    drawing = false;
  });

  canvas.addEventListener('mouseleave', () => {
    drawing = false;
  });

   // Touch events
  // We will treat a single touch like left click (white pen).
  // If the user wants to draw black, they can still do so with right-click on desktop.
  // For mobile, we only implement a single pen color (white) when touching.
  // If black is desired, more advanced UI would be needed (e.g., a toggle button).
  // But for simplicity, we will just draw white on touch.
  
  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    drawing = true;
    // default to white for touch
    currentColor = '#FFFFFF';
    drawPixelFromTouch(e);
  }, { passive: false });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (drawing) drawPixelFromTouch(e);
  }, { passive: false });

  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    drawing = false;
  }, { passive: false });

  canvas.addEventListener('touchcancel', (e) => {
    e.preventDefault();
    drawing = false;
  }, { passive: false });

  function drawPixelFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width; 
    const scaleY = canvas.height / rect.height; 
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);
    ctx.fillStyle = currentColor; 
    ctx.fillRect(x, y, PEN_SIZE, PEN_SIZE);
  }

  function drawPixelFromTouch(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width; 
    const scaleY = canvas.height / rect.height; 
    // We only consider the first touch for drawing
    const touch = e.touches[0];
    const x = Math.floor((touch.clientX - rect.left) * scaleX);
    const y = Math.floor((touch.clientY - rect.top) * scaleY);
    ctx.fillStyle = currentColor; 
    ctx.fillRect(x, y, PEN_SIZE, PEN_SIZE);
  }


  function drawPixel(e) {
    const rect = canvas.getBoundingClientRect();
    // Calculate coordinates relative to the canvas scale
    const scaleX = canvas.width / rect.width; 
    const scaleY = canvas.height / rect.height; 
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);
    ctx.fillStyle = currentColor; 
    ctx.fillRect(x, y, PEN_SIZE, PEN_SIZE);
  }

  document.getElementById('submitBtn').addEventListener('click', () => {
    const dataURL = canvas.toDataURL('image/png');
    fetch('/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    })
	.then(response => response.text())
	.then(text => {
		document.getElementById('antwortAbschnitt').textContent = text;
	})
    .catch(err => {
      console.error(err);
      alert('An error occurred.');
    });
  });

  document.getElementById('clearBtn').addEventListener('click', () => {
    clearCanvas();
  });
</script>
</body>
</html>
`
	w.Write([]byte(html))
}

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

func execModel(input []float64) int {
	// Vorhersage treffen
	digit := predict(input, W1, b1, W2, b2)
	fmt.Printf("Das Modell erkennt die Ziffer als: %d\n", digit)
	return digit
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

func handleUpload(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s %s", r.RemoteAddr, r.Method, r.URL)

	if r.Method != http.MethodPost {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	type payload struct {
		Image string `json:"image"`
	}

	var pl payload
	if err := json.NewDecoder(r.Body).Decode(&pl); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}

	data := pl.Image
	prefix := "data:image/png;base64,"
	if !strings.HasPrefix(data, prefix) {
		http.Error(w, "Invalid image data", http.StatusBadRequest)
		return
	}
	data = strings.TrimPrefix(data, prefix)

	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		http.Error(w, "Decoding base64 failed", http.StatusBadRequest)
		return
	}

	if useImageMagick {
		if err := os.WriteFile("digit_uploaded.png", decoded, 0644); err != nil {
			http.Error(w, "Failed to save file", http.StatusInternalServerError)
			return
		}
		if runtime.GOOS == "darwin" {
			exec.Command("magick", "convert", "digit_uploaded.png", "-colorspace", "Gray", "-separate", "-evaluate-sequence", "Min", "digit.png").Run()
		} else {
			exec.Command("convert", "digit_uploaded.png", "-colorspace", "Gray", "-separate", "-evaluate-sequence", "Min", "digit.png").Run()
		}
	} else {
		grayImage := image.NewGray(image.Rect(0, 0, 28, 28))
		for i, val := range decoded {
			x := i % 28
			y := i / 28
			grayImage.SetGray(x, y, color.Gray{Y: val})
		}

		outFile, err := os.Create("digit.png")
		if err != nil {
			http.Error(w, "Failed to create output file", http.StatusInternalServerError)
			return
		}
		defer outFile.Close()

		if err := png.Encode(outFile, grayImage); err != nil {
			http.Error(w, "Failed to save file", http.StatusInternalServerError)
			return
		}
	}

	w.WriteHeader(http.StatusOK)

	input, err := loadAndPreprocessImage("digit.png")
	if err != nil {
		http.Error(w, "Failed to preprocess image", http.StatusInternalServerError)
		return
	}

	result := execModel(input)
	fmt.Fprintf(w, "Das Modell erkennt die Ziffer als: %d\n", result)
}
