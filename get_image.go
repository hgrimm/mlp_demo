package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
)

// Dieses Programm nimmt einen Index aus den CLI-Argumenten, lädt dieses Bild aus den MNIST Trainingsdaten
// und speichert es als PNG ab. Außerdem wird das zugehörige Label auf der Konsole ausgegeben.

func main() {
	// Index als CLI-Argument einlesen
	var index int
	flag.IntVar(&index, "index", 0, "Index des MNIST-Bildes, das extrahiert werden soll")
	flag.Parse()

	// Wenn kein -index Argument gegeben ist, versuchen wir den ersten CLI-Argument ohne Flag zu nehmen
	if flag.NArg() > 0 {
		var err error
		index, err = atoi(flag.Arg(0))
		if err != nil {
			log.Fatalf("Fehler beim Parsen des Index: %v", err)
		}
	}

	imageFile := "mnist/t10k-images-idx3-ubyte"
	labelFile := "mnist/t10k-labels-idx1-ubyte"

	imgData, err := loadMNISTImage(imageFile, index)
	if err != nil {
		log.Fatalf("Fehler beim Laden des Bildes: %v", err)
	}

	label, err := loadMNISTLabel(labelFile, index)
	if err != nil {
		log.Fatalf("Fehler beim Laden des Labels: %v", err)
	}

	// imgData ist ein []byte mit 784 Pixeln (28x28)
	// Erstellen wir ein Grau-Bild und schreiben es als PNG
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	for i, val := range imgData {
		// val ist ein Byte von 0 bis 255
		x := i % 28
		y := i / 28
		img.SetGray(x, y, color.Gray{Y: val})
	}

	outFile, err := os.Create("digit.png")
	if err != nil {
		log.Fatalf("Fehler beim Erstellen der Ausgabedatei: %v", err)
	}
	defer outFile.Close()

	if err := png.Encode(outFile, img); err != nil {
		log.Fatalf("Fehler beim Schreiben des PNG: %v", err)
	}

	fmt.Printf("Bild mit Index %d wurde als 'digit.png' gespeichert. Label: %d\n", index, label)
}

func atoi(s string) (int, error) {
	var n int
	_, err := fmt.Sscan(s, &n)
	return n, err
}

// loadMNISTImage lädt das Bild mit gegebenem Index aus der MNIST-Bilddatei (train-images-idx3-ubyte)
func loadMNISTImage(filename string, idx int) ([]byte, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, numImages, rows, cols int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &rows); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &cols); err != nil {
		return nil, err
	}

	if idx < 0 || int32(idx) >= numImages {
		return nil, fmt.Errorf("Index außerhalb der Reichweite. Anzahl Bilder: %d", numImages)
	}

	// Zu überspringende Bytes bis zu unserem Bild: idx * rows * cols
	offset := int64(idx) * int64(rows) * int64(cols)
	// Wir müssen den Header (16 Bytes) überspringen, 16 Bytes wurden bereits gelesen durch binary.Read
	// Fileposition nach den Headers: jetzt bei Byte 16, also setzen wir weiter:
	if _, err := f.Seek(16+offset, 0); err != nil {
		return nil, err
	}

	imgData := make([]byte, rows*cols)
	if _, err := f.Read(imgData); err != nil {
		return nil, err
	}

	return imgData, nil
}

// loadMNISTLabel lädt das Label mit gegebenem Index aus der MNIST-Labeldatei (train-labels-idx1-ubyte)
func loadMNISTLabel(filename string, idx int) (uint8, error) {
	f, err := os.Open(filename)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var magic, numLabels int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return 0, err
	}
	if err := binary.Read(f, binary.BigEndian, &numLabels); err != nil {
		return 0, err
	}

	if idx < 0 || int32(idx) >= numLabels {
		return 0, fmt.Errorf("Index außerhalb der Reichweite. Anzahl Labels: %d", numLabels)
	}

	// Header size für labels: 8 Bytes (2x int32)
	// Jedes Label ist 1 Byte
	offset := int64(idx)

	if _, err := f.Seek(8+offset, 0); err != nil {
		return 0, err
	}

	var label uint8
	if err := binary.Read(f, binary.BigEndian, &label); err != nil {
		return 0, err
	}

	return label, nil
}
