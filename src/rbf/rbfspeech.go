/*
Radial Basis Function Neural Network (RBFNN)
This is a web application that uses the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/SpeechSynRBF.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the Neural Network.  Each example consists
of a spectrogram of synthetic speech and a desired class output.  The RBFNN
itself consists of an input layer of nodes, one hidden layer containing nodes,
and an output layer of nodes.  The input to hidden layer nodes are fully
connected but without any weights.  Radial Basis Functions are used as
the hidden layer activation functions.  The RBFs are Gaussian functions with
mean and standard deviation which are determined by K-means clustering.
The hidden to output layer nodes are fully connected by weighted links.
The weights are trained by back propagating the output layer errors forward to the
hidden layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layer.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.  The RBF Neural Network uses the Sigmoid
(Logistic) function in the output layer.  Mean-square error loss is used to compute
the error in the ouput layer with an encoded vector as the target or desired output.
This is a classification problem and only one of the ouputs is one, the rest are zero.
Therefore the outputs are probabilities with values between 0 and 1.

This application classifies synthetic speech patterns.  The spectrogram of
each speech pattern file is calculated and the spectrogram is the input to the RBF.
The RBF classifies the speech pattern based on its spectral content versus time. The test
results are shown.  The user can plot the time domain or the spectrogram
(frequency versus time) of the synthetic speech.  The spectrogram is a three-dimentional
plot of the spectral power versus time.  The third dimension is a grayscale color.
Short-time Fourier Transforms (STFT) are used to compute the FFT from 20-30 ms blocks
of synthetic speech data.

The synthetic speech is generated with a sum of sinusoids (voiced) or gaussian noise (unvoiced) in
20-30 ms frames.  If voiced, the fundamental is randomly chosen from between 200 and 800 Hz. Each voiced
speech has 1-5 subfrequencies with a smaller amplitude than the fundamental.  The amplitudes are randomly
chosen and can be varied.  The duration of each frame can also be varied.  The variation of these parameters
will test the generalization capabilities of the Neural Network.  The testing phase varies the parameters
based upon the user input.  The percentage of correct classification is presented in graphical and tabular
forms upon completion of the testing.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
	"github.com/thomasteplick/cluster"
)

const (
	addr               = "127.0.0.1:8080"             // http server listen address
	fileTrainingRBF    = "templates/trainingRBF.html" // html for training RBF
	fileTestingRBF     = "templates/testingRBF.html"  // html for testing RBF
	fileDisplayRBF     = "templates/displayRBF.html"  // html for speech time or spectrogram plots
	patternTrainingRBF = "/speechRBFtrain"            // http handler for training the RBF
	patternTestingRBF  = "/speechRBFtest"             // http handler for testing the RBF
	patternDisplayRBF  = "/speechRBFdisplay"          // http handler for displaying the RBF
	xlabels            = 11                           // # labels on x axis
	ylabels            = 11                           // # labels on y axis
	fileweights        = "weights.csv"                // rbf weights
	filekmeans         = "kmeans.csv"                 // k-means cluster data:  centroids, bandwidth, wcss
	synSpeech          = "synSpeech.wav"              // synthetic speech wav file
	dataDir            = "data/"                      // directory for the weights and synthetic speech files
	rows               = 300                          // rows in canvas
	cols               = 300                          // columns in canvas
	sampleRate         = 8000                         // Hz or samples/sec
	bitDepth           = 16                           // audio wav encoder/decoder sample size
	ncolors            = 5                            // number of grayscale colors in spectrogram
	nffts              = 64                           // number of ffts in the spectrograms
	avgDuration        = 200                          // average duration in samples of the speech frame size
	classes            = 64                           // number of classes is the number of speech patterns or words
	maxSubFreq         = 5                            // max number of sub-frequencies
	a                  = 1.7159                       // activation function const
	b                  = 2.0 / 3.0                    // activation function const
	K1                 = b / a
	K2                 = a * a
)

// test statistics that are tabulated in HTML
type Results struct {
	Class   string // int
	Correct string // int      percent correct
	Count   string // int      number of training examples in the class
}

// Type to contain all the HTML template actions
type PlotT struct {
	Grid          []string  // plotting grid
	Status        string    // status of the plot
	Xlabel        []string  // x-axis labels
	Ylabel        []string  // y-axis labels
	HiddenLayers  string    // number of hidden layers
	LayerDepth    string    // number of Nodes in hidden layers
	LearningRate  string    // size of weight update for each iteration
	Momentum      string    // previous weight update scaling factor
	Epochs        string    // number of epochs
	FFTSize       string    // 8192, 4098, 2048, 1024
	FFTWindow     string    // Bartlett, Welch, Hamming, Hanning, Rectangle
	Domain        string    // plot time or spectrogra domain
	TestResults   []Results // tabulated statistics of testing
	TotalCount    string    // Results tabulation
	TotalCorrect  string
	DelDuration   string // delta of the speech frame duration
	DelPitch      string // delta of the speech frame frequencies
	DelAmpl       string // delta of the speech frame frequency amplitudes
	PercentVoiced string // percentage of the speech frames voiced
	Classes       string // number of speech patterns
	SpeechPattern string // speech pattern to display
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// graph links used to connect last Feature Map layer to output layer
type Link struct {
	wgt      float64 // weight
	wgtDelta float64 // previous weight update used in momentum
}

type Stats struct {
	correct    []int // % correct classifcation
	classCount []int // #samples in each class
}

// training examples
type Sample struct {
	desired int    // numerical class of the synthetic speech pattern
	data    []Node //  frequency bins from the STFT and deltas from backprop
}

// Speech frame attributes
type SpeechFrame struct {
	freqs []float64
	amps  []float64
}

// K-means protype
type Kmeans struct {
	mean []float64
	bw   float64 // bandwidth
	wcss float64 // within-class squared sum
}

// Primary data structure for holding the RBF state
type RBF struct {
	plot          *PlotT          // data to be distributed in the HTML template
	Endpoints                     // embedded struct
	link          [][]Link        // links in the graph
	node          [][]Node        // nodes in graph
	nsamples      int             // number of synthetic speech pattern
	domain        string          // time or spectrogram plot
	mse           []float64       // mean square error in output layer per epoch used in Learning Curve
	epochs        int             // number of epochs
	learningRate  float64         // learning rate parameter
	momentum      float64         // delta weight scale constant
	hiddenLayers  int             // number of hidden layers
	desired       []float64       // desired output of the sample
	layerDepth    int             // hidden layer number of nodes
	words         []string        // classified words in test message
	grayscale     map[int]string  // grayscale for spectrogram
	fftSize       int             // FFT size for spectrogram
	fftWindow     string          // FFT window
	speechPat     [][]SpeechFrame // speech pattern
	delPitch      int             // pitch delta
	delDuration   int             // duration delta in samples
	delAmpl       float64         // amplitude delta of the frequencies
	percentVoiced int             // percent voiced
	synSpeech     []float64       // synthetic speech
	statistics    Stats
	freqs         []float64 // speech frame frequencies
	amps          []float64 // speech frame amplitudes
	cluster       []Kmeans  // prototype state
}

// Window function type
type Window func(n int, m int) complex128

// global variables for parse and execution of the html template
var (
	tmplTrainingRBF *template.Template
	tmplTestingRBF  *template.Template
	tmplDisplayRBF  *template.Template
	winType         = []string{"Bartlett", "Welch", "Hamming", "Hanning", "Rectangle"}
)

// init parses the html template files
func init() {
	tmplTrainingRBF = template.Must(template.ParseFiles(fileTrainingRBF))
	tmplTestingRBF = template.Must(template.ParseFiles(fileTestingRBF))
	tmplDisplayRBF = template.Must(template.ParseFiles(fileDisplayRBF))
}

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// calculateMSE calculates the MSE at the output layer every epoch
func (rbf *RBF) calculateMSE(epoch int) {
	// loop over the output layer nodes
	err := 0.0
	outputLayer := rbf.hiddenLayers + 1
	//fmt.Printf("epoch=%d:", epoch)
	for n := 0; n < len(rbf.node[outputLayer]); n++ {
		// Calculate (desired[n] - rbf.node[L][n].y)^2 and store in rbf.mse[n]
		err = rbf.desired[n] - rbf.node[outputLayer][n].y
		err2 := err * err
		rbf.mse[epoch] += err2
		//fmt.Printf("%.3f - %.3f ", rbf.desired[n], rbf.node[outputLayer][n].y)
	}
	//fmt.Println()
	rbf.mse[epoch] /= float64(len(rbf.node[outputLayer]))

	// calculate min/max mse
	if rbf.mse[epoch] < rbf.ymin {
		rbf.ymin = rbf.mse[epoch]
	}
	if rbf.mse[epoch] > rbf.ymax {
		rbf.ymax = rbf.mse[epoch]
	}
}

// determineClass determines testing example class given sample number and sample
func (rbf *RBF) determineClass(sample *Sample) error {
	// At output layer, classify example and increment class/correct count
	// convert node outputs to the class; zero is the threshold
	class := 0
	for i, output := range rbf.node[rbf.hiddenLayers+1] {
		if output.y > 0.0 {
			class |= (1 << i)
		}
	}

	// Assign Stats.correct, Stats.classCount
	rbf.statistics.classCount[sample.desired]++
	if class == sample.desired {
		rbf.statistics.correct[class]++
	}

	return nil
}

// class2desired constructs the desired output from the given class
func (rbf *RBF) class2desired(class int) {
	// tranform int to slice of -1 and 1 representing the 0 and 1 bits
	for i := 0; i < len(rbf.desired); i++ {
		if class&1 == 1 {
			rbf.desired[i] = 1
		} else {
			rbf.desired[i] = -1
		}
		class >>= 1
	}
}

// Gauss RBF with centroid and bandwidth
func (rbf *RBF) gaussRBF(x []float64, mean []float64, bw float64) float64 {
	// sigma = bw
	sum := 0.0
	for i := range x {
		diff := x[i] - mean[i]
		sum += diff * diff
	}
	return math.Exp(-sum / (2.0 * bw * bw))
}

func (rbf *RBF) propagateForward(samp *Sample) error {
	// Assign sample to input layer
	layer := 0
	v := make([]float64, len(rbf.node[layer]))
	for i, val := range samp.data {
		rbf.node[layer][i].y = val.y
		v[i] = val.y
	}

	// calculate desired from the class
	rbf.class2desired(samp.desired)

	// Loop over layers: input + hiddenLayer + output layer
	// input->hidden, then hidden->output
	maxVal := 0.0
	for layer := 1; layer <= rbf.hiddenLayers; layer++ {
		// Loop over nodes in the layer, d1 is the layer depth of current
		d1 := len(rbf.node[layer])
		for i1 := range d1 { // current layer loop
			// The network is fully connected.
			// compute output y = RBF output
			rbf.node[layer][i1].y = rbf.gaussRBF(v, rbf.cluster[i1].mean, rbf.cluster[i1].bw)
			if rbf.node[layer][i1].y > maxVal {
				maxVal = rbf.node[layer][i1].y
			}
		}
		// normalize RBF ouput to be in linear range of output activation function tanh
		for i1 := range d1 {
			rbf.node[layer][i1].y = 2.0 * (rbf.node[layer][i1].y/maxVal - 0.5)
		}
	}

	// Loop over nodes hidden layer to output layer
	layer = rbf.hiddenLayers + 1
	d1 := len(rbf.node[layer])
	for i1 := range d1 { // current layer loop
		// Each node in previous layer is connected to current node because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(rbf.node[layer-1])
		v := 0.0
		for i2 := range d2 { // previous layer loop
			v += rbf.link[layer-1][i2*d1+i1].wgt * rbf.node[layer-1][i2].y
		}
		// compute output y = Phi(v)
		rbf.node[layer][i1].y = a * math.Tanh(b*v)
	}
	return nil
}

func (rbf *RBF) propagateBackward() error {

	// Loop over nodes in output layer to hidden layer
	layer := rbf.hiddenLayers + 1
	d1 := len(rbf.node[layer])
	for i1 := range d1 { // this layer loop
		//compute error e=d-Phi(v)
		rbf.node[layer][i1].delta = rbf.desired[i1] - rbf.node[layer][i1].y
		// Multiply error by this node's Phi'(v) to get local gradient.
		rbf.node[layer][i1].delta *= K1 * (K2 - rbf.node[layer][i1].y*rbf.node[layer][i1].y)
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(rbf.node[layer-1])
		for i2 := range d2 { // previous layer loop
			// Compute weight delta, Update weight with momentum, y, and local gradient
			wgtDelta := rbf.learningRate * rbf.node[layer][i1].delta * rbf.node[layer-1][i2].y
			rbf.link[layer-1][i2*d1+i1].wgt +=
				wgtDelta + rbf.momentum*rbf.link[layer-1][i2*d1+i1].wgtDelta
			// update weight delta
			rbf.link[layer-1][i2*d1+i1].wgtDelta = wgtDelta
		}
		// Reset this local gradient to zero for next training example
		rbf.node[layer][i1].delta = 0.0
	}

	return nil
}

// runTrainingEpochs performs forward and backward propagation over each sample
func (rbf *RBF) runTrainingEpochs() error {

	// Initialize the weights

	// input layer, direct connection to input data
	// initialize the wgt to one, wgtDelta = 0
	for i := range rbf.link[0] {
		rbf.link[0][i].wgt = 1.0
		rbf.link[0][i].wgtDelta = 0.0
	}

	// output layer links
	for i := range rbf.link[rbf.hiddenLayers] {
		rbf.link[rbf.hiddenLayers][i].wgt = 2.0 * (rand.Float64() - .5) / float64(rbf.layerDepth)
		rbf.link[rbf.hiddenLayers][i].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(rbf.layerDepth)
		//rbf.link[rbf.hiddenLayers][i].wgt = 2.0 * (rand.Float64() - .5)
		//rbf.link[rbf.hiddenLayers][i].wgtDelta = 2.0 * (rand.Float64() - .5)
	}

	// Create a sample for containing the spectrogram to propagate forward
	samp := Sample{data: make([]Node, nffts)}

	for n := 0; n < rbf.epochs; n++ {
		// create speech for one pattern
		// Randomly choose a pattern
		pattern := rand.Intn(classes)
		err := rbf.createSpeech(pattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = rbf.createSpeech(pattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				return fmt.Errorf("createSpeech error: %v", err.Error())
			}
		}

		samp.desired = pattern

		// create spectrogram
		err = rbf.createSpectrogram(&samp)
		if err != nil {
			fmt.Printf("createSpectrogram error: %v\n", err.Error())
			return fmt.Errorf("createSpectrogram error: %v", err.Error())
		}

		// Forward Propagation
		err = rbf.propagateForward(&samp)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}

		// Backward Propagation
		err = rbf.propagateBackward()
		if err != nil {
			return fmt.Errorf("backward propagation error: %s", err.Error())
		}

		//fmt.Printf("epoch=%d\n", n)
		// At the end of each epoch, loop over the output nodes and calculate mse
		rbf.calculateMSE(n)
		//fmt.Println()

	}
	return nil
}

// createSpectrogram creates spectrograms from the synthetic speech used in runEpochs
func (rbf *RBF) createSpectrogram(samp *Sample) error {

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD := make([]float64, rbf.fftSize/2)

	rbf.nsamples = rbf.fftSize * nffts

	// Create the spectrogram, no overlap
	// loop over the samples, with fftSize jump
	i := 0
	for smpl := 0; smpl < len(rbf.synSpeech); smpl += rbf.fftSize {
		i = smpl / rbf.fftSize
		binMax, _, err := rbf.calculatePSD(rbf.synSpeech[smpl:smpl+rbf.fftSize], PSD, rbf.fftWindow, rbf.fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}
		samp.data[i].y = float64(binMax)
	}
	return nil
}

// create synthetic speech patterns consisting of 25ms frames of voiced or unvoiced input
func (rbf *RBF) createPatterns() error {
	nsamples := rbf.fftSize * nffts
	// a block consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
	const (
		amplMinf1  = 500.0
		amplMaxf1  = 1000.0
		f1Min      = 100   // Hz = cycles/sec
		f1Max      = 3800  // Hz = cycles/sec
		sigmaNoise = 200.0 // unvoiced speech
		nyquist    = sampleRate / 2
		frameScale = 2 // frame extender
	)

	nsubfreq := 0

	// make SpeechFrames for the speech patterns
	rbf.speechPat = make([][]SpeechFrame, classes)
	for i := range rbf.speechPat {
		rbf.speechPat[i] = make([]SpeechFrame, nframes)
	}

	// Create the speech patterns for voiced or unvoiced frames
	// loop over number of speech patterns
	for pat := 0; pat < classes; pat++ {
		speechfile := filepath.Join(dataDir, fmt.Sprintf("speech%d.csv", pat))
		fspeech, err := os.Create(speechfile)
		if err != nil {
			return fmt.Errorf("createPatterns could not create file %s error: %s", speechfile, err.Error())
		}
		// loop over the number of frames
		for fr := 0; fr < nframes; fr++ {
			// select frequencies and amplitudes depending on voiced/unvoiced
			if fr%frameScale == 0 {
				if 100.0*rand.Float64() < float64(rbf.percentVoiced) {
					// voiced, cycles/sec = Hz
					// 1-5 subfrequencies
					nsubfreq = rand.Intn(maxSubFreq) + 1
					rbf.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
					rbf.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
					// fundamental frequency
					rbf.speechPat[pat][fr].freqs[0] = f1Min + (f1Max-f1Min)*rand.Float64()
					rbf.speechPat[pat][fr].amps[0] = amplMinf1 + (amplMaxf1-amplMinf1)*rand.Float64()
					// subfrequencies
					for sf := 0; sf < nsubfreq; sf++ {
						rbf.speechPat[pat][fr].freqs[sf+1] =
							rbf.speechPat[pat][fr].freqs[sf] + (nyquist-rbf.speechPat[pat][fr].freqs[sf])*rand.Float64()
						rbf.speechPat[pat][fr].amps[sf+1] = rbf.speechPat[pat][fr].amps[sf] * 0.9
					}
				} else {
					// unvoiced is gaussian noise
					nsubfreq = 0
					rbf.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
					rbf.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
					rbf.speechPat[pat][fr].freqs[0] = 0.0
					rbf.speechPat[pat][fr].amps[0] = sigmaNoise * rand.Float64()
				}
			} else {
				// Use previous frame parameters so that the frame is extended
				rbf.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
				rbf.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
				// fundamental frequency
				rbf.speechPat[pat][fr].freqs[0] = rbf.speechPat[pat][fr-1].freqs[0]
				rbf.speechPat[pat][fr].amps[0] = rbf.speechPat[pat][fr-1].amps[0]
				// subfrequencies
				for sf := 0; sf < nsubfreq; sf++ {
					rbf.speechPat[pat][fr].freqs[sf+1] = rbf.speechPat[pat][fr-1].freqs[sf+1]
					rbf.speechPat[pat][fr].amps[sf+1] = rbf.speechPat[pat][fr-1].amps[sf+1]
				}
			}
			// Save the speech frame to disk file
			// frequencies
			for _, freq := range rbf.speechPat[pat][fr].freqs {
				_, err = fmt.Fprintf(fspeech, "%.16f,", freq)
				if err != nil {
					return fmt.Errorf("createPatterns file: %s, frame: %d, freq: %f write error: %s", speechfile, fr, freq, err.Error())
				}
			}
			// amplitudes
			for _, amp := range rbf.speechPat[pat][fr].amps[0 : len(rbf.speechPat[pat][fr].amps)-1] {
				_, err = fmt.Fprintf(fspeech, "%.16f,", amp)
				if err != nil {
					return fmt.Errorf("createPatterns file: %s, frame: %d, ampl: %f write error: %s", speechfile, fr, amp, err.Error())
				}
			}
			_, err = fmt.Fprintf(fspeech, "%.16f\n", rbf.speechPat[pat][fr].amps[len(rbf.speechPat[pat][fr].amps)-1])
			if err != nil {
				return fmt.Errorf("createPatterns file: %s, frame: %d write error: %s", speechfile, fr, err.Error())
			}
		}
		fspeech.Close()
	}
	return nil
}

// Create K-means cluster data:  centroids, bandwidth, and WCSS
func (rbf *RBF) createKmeansCluster() error {
	const kMeansSamples = 100
	data := make([][]float64, kMeansSamples*classes)
	samp := Sample{data: make([]Node, nffts)}
	for i := range data {
		data[i] = make([]float64, nffts)
	}
	for i := range kMeansSamples {
		for word := range classes {
			// create speech for the word
			err := rbf.createSpeech(word)
			for err != nil {
				if err.Error() == "repeat" {
					err = rbf.createSpeech(word)
				} else {
					fmt.Printf("createSpeech error: %v\n", err.Error())
					return fmt.Errorf("createSpeech error: %v", err.Error())
				}
			}

			// create spectrogram
			err = rbf.createSpectrogram(&samp)
			if err != nil {
				fmt.Printf("createSpectrogram error: %v\n", err.Error())
				return fmt.Errorf("createSpectrogram error: %v", err.Error())
			}

			// copy the spectrogram data
			for j, node := range samp.data {
				data[i*classes+word][j] = node.y
			}
		}
	}

	// create the K-means
	err := cluster.Kmeans(rbf.layerDepth, data)
	if err != nil {
		fmt.Printf("cluster.Kmeans error: %v\n", err.Error())
		return fmt.Errorf("cluster.Kmeans error: %v", err.Error())
	}
	return nil
}

// Retrieve K-means cluster data:  centroids, bandwidth, and WCSS
func (rbf *RBF) getKmeansCluster(endpoints *Endpoints) error {
	fKmeans, err := os.Open(path.Join(dataDir, filekmeans))
	if err != nil {
		fmt.Printf("Open file %s error: %v", filekmeans, err)
		return fmt.Errorf("open file %s error: %s", filekmeans, err.Error())
	}
	defer fKmeans.Close()

	scanner := bufio.NewScanner(fKmeans)
	n := 0
	// save min and max of wcss
	endpoints.ymin = math.MaxFloat64
	endpoints.ymax = 0.0
	for scanner.Scan() {
		line := scanner.Text()
		items := strings.Split(line, ",")
		if len(items) != (nffts + 2) {
			fmt.Printf("getKmeansCluster, len(items) = %d, should be %d\n", len(items), nffts+2)
			return fmt.Errorf("getKmeansCluster, len(items) = %d, should be %d", len(items), nffts+2)
		}
		for i := range nffts {
			bin, err := strconv.ParseFloat(items[i], 64)
			if err != nil {
				fmt.Printf("ParseFloat in getKmeansCluster bin error: %v\n", err.Error())
				continue
			}
			rbf.cluster[n].mean[i] = bin
		}
		bw, err := strconv.ParseFloat(items[nffts], 64)
		if err != nil {
			fmt.Printf("ParseFloat in getKmeansCluster bw error: %v\n", err.Error())
			continue
		}
		rbf.cluster[n].bw = bw
		wcss, err := strconv.ParseFloat(items[nffts+1], 64)
		if err != nil {
			fmt.Printf("ParseFloat in getKmeansCluster wcss error: %v\n", err.Error())
			continue
		}
		rbf.cluster[n].wcss = wcss
		if wcss > endpoints.ymax {
			endpoints.ymax = wcss
		}
		if wcss < endpoints.ymin {
			endpoints.ymin = wcss
		}
		n++
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("getKmeansCluster scanner error: %s\n", err.Error())
		return fmt.Errorf("getKmeansCluster scanner error: %v", err)
	}

	return nil
}

// synthesize creates synthetic speech using frequencies and amplitudes of sinusoids or gaussian noise
func (rbf *RBF) synthesize(nfreqs int, start int, stop int) error {

	t := 0.0
	step := 1.0 / float64(sampleRate)
	var sum float64
	// calculate speech over the interval
	if start == 0 {
		sum = 0.0
		if nfreqs == 1 {
			sum += rbf.amps[0] * rand.NormFloat64()
		} else {
			for j := 0; j < nfreqs; j++ {
				sum += rbf.amps[j] * math.Sin(2.0*math.Pi*rbf.freqs[j]*t)
			}
		}
		t += step
		rbf.synSpeech[0] = sum
		start++
	}
	for i := start; i < stop; i++ {
		sum = 0.0
		if nfreqs == 1 {
			sum += rbf.amps[0] * rand.NormFloat64()
		} else {
			for j := 0; j < nfreqs; j++ {
				sum += rbf.amps[j] * math.Sin(2.0*math.Pi*rbf.freqs[j]*t)
			}
		}
		t += step
		rbf.synSpeech[i] = 0.5 * (sum + rbf.synSpeech[i-1])
	}
	return nil
}

// createSpeech creates a slice of training/testing synthetic speech based on the speech patterns
func (rbf *RBF) createSpeech(pattern int) error {
	// use the speech patterns and apply deltas for the frequencies, amplitude, and duration so the RBF generalizes
	nsamples := rbf.fftSize * nffts
	remain := nsamples
	nframes := len(rbf.speechPat[pattern])
	// starting sample for current frame
	samp := 0

	const (
		margin1       int = avgDuration - 40
		margin2       int = 2 * margin1
		delSigmaNoise     = 1.0
	)

	// loop over the frame of the pattern and retrieve the attributes of the speech frame
	// add delta for pitch and duration
	for frame := 0; frame < nframes-1; frame++ {
		if remain < margin2 {
			return fmt.Errorf("repeat")
		}
		nfreqs := len(rbf.speechPat[pattern][frame].freqs)
		// unvoiced pattern, frequency = 0
		if nfreqs == 1 {
			if rand.Intn(2) > 0 {
				rbf.amps[0] = rbf.speechPat[pattern][frame].amps[0] + delSigmaNoise*rbf.delAmpl
			} else {
				rbf.amps[0] = rbf.speechPat[pattern][frame].amps[0] - delSigmaNoise*rbf.delAmpl
			}
			rbf.freqs[0] = rbf.speechPat[pattern][frame].freqs[0]
		} else {
			// voiced pattern
			for i := 0; i < nfreqs; i++ {
				if rand.Intn(2) > 0 {
					rbf.freqs[i] =
						rbf.speechPat[pattern][frame].freqs[i] + float64(rbf.delPitch)
				} else {
					rbf.freqs[i] =
						rbf.speechPat[pattern][frame].freqs[i] - float64(rbf.delPitch)/rbf.speechPat[pattern][frame].freqs[i]
				}

				if rand.Intn(2) > 0 {
					rbf.amps[i] = rbf.speechPat[pattern][frame].amps[i] * (1.0 + float64(rbf.delAmpl))
				} else {
					rbf.amps[i] = rbf.speechPat[pattern][frame].amps[i] * (1.0 - float64(rbf.delAmpl))
				}
			}
		}
		duration := avgDuration
		// add or subtract delta
		if rand.Intn(2) > 0 {
			duration += rbf.delDuration
		} else {
			duration -= rbf.delDuration
		}
		remain -= duration

		err := rbf.synthesize(nfreqs, samp, samp+duration)
		if err != nil {
			return fmt.Errorf("synthesize error: %v", err.Error())
		}
		samp += duration
	}

	if remain < margin1 {
		return fmt.Errorf("repeat")
	}

	nfreqs := len(rbf.speechPat[pattern][nframes-1].freqs)
	// unvoiced pattern, frequency = 0
	if nfreqs == 1 {
		if rand.Intn(2) > 0 {
			rbf.amps[0] = rbf.speechPat[pattern][nframes-1].amps[0] + delSigmaNoise*rbf.delAmpl
		} else {
			rbf.amps[0] = rbf.speechPat[pattern][nframes-1].amps[0] - delSigmaNoise*rbf.delAmpl
		}
		rbf.freqs[0] = rbf.speechPat[pattern][nframes-1].freqs[0]
	} else {
		// voiced pattern
		for i := 1; i < nfreqs; i++ {
			if rand.Intn(2) > 0 {
				rbf.freqs[i] =
					rbf.speechPat[pattern][nframes-1].freqs[i] + float64(rbf.delPitch)/rbf.speechPat[pattern][nframes-1].freqs[i]
			} else {
				rbf.freqs[i] =
					rbf.speechPat[pattern][nframes-1].freqs[i] - float64(rbf.delPitch)/rbf.speechPat[pattern][nframes-1].freqs[i]
			}

			if rand.Intn(2) > 0 {
				rbf.amps[i] = rbf.speechPat[pattern][nframes-1].amps[i] * (1.0 + float64(rbf.delAmpl))
			} else {
				rbf.amps[i] = rbf.speechPat[pattern][nframes-1].amps[i] * (1.0 - float64(rbf.delAmpl))
			}
		}
	}

	err := rbf.synthesize(nfreqs, samp, samp+remain)
	if err != nil {
		return fmt.Errorf("synthesize error: %v", err.Error())
	}

	return nil
}

// newTrainingRBF constructs an RBF instance for training
func newTrainingRBF(r *http.Request, hiddenLayers int, plot *PlotT) (*RBF, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("layerdepth")
	layerDepth, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("layerdepth int conversion error: %v\n", err)
		return nil, fmt.Errorf("layerdepth int conversion error: %s", err.Error())
	}

	txt = r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	txt = r.FormValue("momentum")
	momentum, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("momentum float conversion error: %v\n", err)
		return nil, fmt.Errorf("momentum float conversion error: %s", err.Error())
	}

	txt = r.FormValue("epochs")
	epochs, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("epochs int conversion error: %v\n", err)
		return nil, fmt.Errorf("epochs int conversion error: %s", err.Error())
	}

	fftWindow := r.FormValue("fftwindow")

	txt = r.FormValue("fftsize")
	fftSize, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("fftsize int conversion error: %v\n", err)
		return nil, err
	}

	// Get delta pitch, delta duration, delta amplitude, and percent voiced speech
	txt = r.FormValue("delpitch")
	delPitch, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delta Pitch conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("delduration")
	delDuration, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delta Duration conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("delampl")
	delAmpl, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("delta Amplitude conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("percentvoiced")
	percentVoiced, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("percent Voiced conversion error: %v\n", err)
		return nil, err
	}

	rbf := RBF{
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		epochs:       epochs,
		learningRate: learningRate,
		momentum:     momentum,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		words:         make([]string, 0),
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		delDuration:   delDuration,
		delPitch:      delPitch,
		delAmpl:       delAmpl,
		percentVoiced: percentVoiced,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
		cluster:       make([]Kmeans, layerDepth),
	}
	// make SpeechFrames for the speech patterns
	nsamples := rbf.fftSize * nffts

	// outer layer nodes
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// input layer nodes are largest PSD bins for each STFT
	ilnodes := classes

	// construct link that holds the weights and weight deltas
	rbf.link = make([][]Link, hiddenLayers+1)

	// input layer
	rbf.link[0] = make([]Link, ilnodes*layerDepth)

	// output layer links
	rbf.link[len(rbf.link)-1] = make([]Link, olnodes*layerDepth)

	// construct nodes
	rbf.node = make([][]Node, hiddenLayers+2)

	// input layer
	rbf.node[0] = make([]Node, ilnodes)

	// output layer
	rbf.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layer
	for i := 1; i <= hiddenLayers; i++ {
		rbf.node[i] = make([]Node, layerDepth)
	}

	// mean-square error
	rbf.mse = make([]float64, epochs)

	// synthetic speech for creating speech with synthesize
	rbf.synSpeech = make([]float64, nsamples)

	// K-means cluster data for centroids
	for i := range layerDepth {
		rbf.cluster[i].mean = make([]float64, ilnodes)
	}

	// construct desired from classes, binary representation
	rbf.desired = make([]float64, olnodes)

	return &rbf, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (rbf *RBF) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = rbf.mse[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (rbf.xmax - rbf.xmin)
	yscale = float64(rows-1) / (rbf.ymax - rbf.ymin)

	rbf.plot.Grid = make([]string, rows*cols)

	// This cell location (row,col) is on the line
	row := int((rbf.ymax-y)*yscale + .5)
	col := int((x-rbf.xmin)*xscale + .5)
	rbf.plot.Grid[row*cols+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := rbf.ymax - rbf.ymin
	lenEPx := rbf.xmax - rbf.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < len(rbf.mse); i++ {
		x++
		// mse/epoch
		y = rbf.mse[i]

		// This cell location (row,col) is on the line
		row := int((rbf.ymax-y)*yscale + .5)
		col := int((x-rbf.xmin)*xscale + .5)
		rbf.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cols) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(rows) * lenEdgeY / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsY, ncellsX)

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((rbf.ymax-interpY)*yscale + .5)
			col := int((interpX-rbf.xmin)*xscale + .5)
			rbf.plot.Grid[row*cols+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (rbf *RBF) insertLabels() {
	rbf.plot.Xlabel = make([]string, xlabels)
	rbf.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (rbf.xmax - rbf.xmin) / (xlabels - 1)
	x := rbf.xmin
	// First label is empty for alignment purposes
	for i := range rbf.plot.Xlabel {
		rbf.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (rbf.ymax - rbf.ymin) / (ylabels - 1)
	y := rbf.ymin
	for i := range rbf.plot.Ylabel {
		rbf.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingRBF(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		rbf  *RBF
	)

	// Get the number of hidden layers
	txt := r.FormValue("hiddenlayers")
	// Need hidden layers to continue
	if len(txt) > 0 {
		hiddenLayers, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Hidden Layers int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Hidden Layers conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create RBF instance to hold state
		rbf, err = newTrainingRBF(r, hiddenLayers, &plot)
		if err != nil {
			fmt.Printf("newTrainingRBF() error: %v\n", err)
			plot.Status = fmt.Sprintf("newTrainingRBF() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create new synthetic speech patterns and RBF prototypes
		newPattern := r.FormValue("speechpattern")
		if newPattern == "new" {
			if err = rbf.createPatterns(); err != nil {
				fmt.Printf("createPatterns() error: %v\n", err)
				plot.Status = fmt.Sprintf("createPatterns() error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplTrainingRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}

			// Create the K-means cluster data consisting of RBF prototypes
			if err = rbf.createKmeansCluster(); err != nil {
				fmt.Printf("createKmeansCluster error: %v\n", err)
				plot.Status = fmt.Sprintf("createKmeansCluster error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplTrainingRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}

		// Read the synthetic speech patterns
		files, err := os.ReadDir(dataDir)
		if err != nil {
			fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
			plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		if len(files) == 0 {
			fmt.Printf("No synthetic speech files in %s\n", dataDir)
			plot.Status = fmt.Sprintf("No filter files in %s", dataDir)
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		} else {
			// make SpeechFrames for the speech patterns
			nsamples := rbf.fftSize * nffts
			// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
			nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
			rbf.speechPat = make([][]SpeechFrame, classes)
			for i := range rbf.speechPat {
				rbf.speechPat[i] = make([]SpeechFrame, nframes)
			}
			pattern := 0
			// Retrieve the speech files
			for _, dirEntry := range files {
				name := dirEntry.Name()
				if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
					fspeech, err := os.Open(filepath.Join(dataDir, name))
					if err != nil {
						fmt.Printf("Open %s error: %v\n", name, err)
						plot.Status = fmt.Sprintf("Open %s error: %v", name, err.Error())
						// Write to HTTP using template and grid
						if err := tmplTrainingRBF.Execute(w, plot); err != nil {
							log.Fatalf("Write to HTTP output using template with error: %v\n", err)
						}
						return
					}
					scanner := bufio.NewScanner(fspeech)
					frame := 0
					for scanner.Scan() {
						line := scanner.Text()
						items := strings.Split(line, ",")
						nfreqs := len(items) / 2
						rbf.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
						rbf.speechPat[pattern][frame].amps = make([]float64, nfreqs)
						for i := 0; i < nfreqs; i++ {
							freq, err := strconv.ParseFloat(items[i], 64)
							if err != nil {
								plot.Status = fmt.Sprintf("freq %d conversion error: %v", i, err.Error())
								// Write to HTTP using template and grid
								if err := tmplTrainingRBF.Execute(w, plot); err != nil {
									log.Fatalf("Write to HTTP output using template with error: %v\n", err)
								}
								return
							}
							ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
							if err != nil {
								plot.Status = fmt.Sprintf("ampl %d conversion error: %v", i, err.Error())
								// Write to HTTP using template and grid
								if err := tmplTrainingRBF.Execute(w, plot); err != nil {
									log.Fatalf("Write to HTTP output using template with error: %v\n", err)
								}
								return
							}
							rbf.speechPat[pattern][frame].amps[i] = ampl
							rbf.speechPat[pattern][frame].freqs[i] = freq
						}
						frame++
					}
					fspeech.Close()
					if err = scanner.Err(); err != nil {
						fmt.Printf("speech file scanner error: %s", err.Error())
						// Write to HTTP using template and grid
						if err := tmplTrainingRBF.Execute(w, plot); err != nil {
							log.Fatalf("Write to HTTP output using template with error: %v\n", err)
						}
						return
					}
					pattern++
				}
			}
		}

		// retrieve K-means cluster data consisting of RBF prototypes
		if err := rbf.getKmeansCluster(&Endpoints{}); err != nil {
			fmt.Printf("getKmeansCluster error: %v\n", err)
			plot.Status = fmt.Sprintf("getKmeansCluster error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the Epochs
		err = rbf.runTrainingEpochs()
		if err != nil {
			fmt.Printf("runTrainingEpochs() error: %v\n", err)
			plot.Status = fmt.Sprintf("runTrainingEpochs() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put MSE vs Epoch in PlotT
		err = rbf.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		rbf.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		rbf.plot.HiddenLayers = strconv.Itoa(rbf.hiddenLayers)
		rbf.plot.LayerDepth = strconv.Itoa(rbf.layerDepth)
		rbf.plot.LearningRate = strconv.FormatFloat(rbf.learningRate, 'f', 5, 64)
		rbf.plot.Momentum = strconv.FormatFloat(rbf.momentum, 'f', 5, 64)
		rbf.plot.Epochs = strconv.Itoa(rbf.epochs)
		rbf.plot.DelDuration = strconv.Itoa(rbf.delDuration)
		rbf.plot.DelPitch = strconv.Itoa(rbf.delPitch)
		rbf.plot.PercentVoiced = strconv.Itoa(rbf.percentVoiced)
		rbf.plot.DelAmpl = strconv.FormatFloat(rbf.delAmpl, 'f', 3, 64)

		// Save hidden layers, hidden layer depth, classes, epochs, fft size, fft window,
		// window, percent voiced, delta duration, delta pitch and Filters/weights to csv file
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()
		// save RBF parameters
		fmt.Fprintf(f, "%d,%d,%d,%f,%f,%d,%s,%d,%d,%d,%f\n",
			rbf.epochs, rbf.hiddenLayers, rbf.layerDepth, rbf.learningRate, rbf.momentum, rbf.fftSize,
			rbf.fftWindow, rbf.delPitch, rbf.delDuration, rbf.percentVoiced, rbf.delAmpl)

		// save weights, layer by layer
		// save first layer, one weight per line because too long to scan in
		for _, node := range rbf.link[0] {
			fmt.Fprintf(f, "%.16f\n", node.wgt)
		}
		// save output layer with csv, one weight per line because too long to scan in
		for _, layer := range rbf.link[1:] {
			for _, node := range layer {
				fmt.Fprintf(f, "%.16f\n", node.wgt)
			}
		}

		rbf.plot.Status = "Mean-square Error plotted"

		// Execute data on HTML template
		if err = tmplTrainingRBF.Execute(w, rbf.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter RBF Neural Network training parameters."
		// Write to HTTP using template and grid
		if err := tmplTrainingRBF.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}
}

// Welch's Method and Bartlett's Method variation of the Periodogram
func (rbf *RBF) calculatePSD(audio []float64, PSD []float64, fftWindow string, fftSize int) (int, float64, error) {

	N := fftSize
	m := N / 2

	// map of window functions
	window := make(map[string]Window, len(winType))
	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	w, ok := window[fftWindow]
	if !ok {
		fmt.Printf("Invalid FFT window type: %v\n", fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", fftWindow)
	}

	bufN := make([]complex128, N)

	for j := 0; j < len(audio); j++ {
		bufN[j] = complex(audio[j], 0)
	}

	// zero-pad the remaining samples
	for i := len(audio); i < N; i++ {
		bufN[i] = 0
	}

	// window the N samples with chosen window
	for k := 0; k < N; k++ {
		bufN[k] *= w(k, m)
	}

	// Perform N-point complex FFT and add squares to previous values in PSD
	fourierN := fft.FFT(bufN)
	x := cmplx.Abs(fourierN[0])
	PSD[0] = x * x
	psdMax := PSD[0]
	binMax := 0
	for j := 1; j < m; j++ {
		// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
		xj := cmplx.Abs(fourierN[j])
		xNj := cmplx.Abs(fourierN[N-j])
		PSD[j] = xj*xj + xNj*xNj
		if PSD[j] > psdMax {
			psdMax = PSD[j]
			binMax = j
		}
	}

	return binMax, psdMax, nil
}

// runTestingEpochs classifies test examples and tabulates test results
func (rbf *RBF) runTestingEpochs() error {

	// Create a sample for containing the spectrogram to propagate forward
	samp := Sample{data: make([]Node, nffts)}
	rbf.statistics =
		Stats{correct: make([]int, classes), classCount: make([]int, classes)}

	for n := 0; n < rbf.epochs; n++ {
		// create speech for one pattern
		// Randomly choose a pattern
		pattern := rand.Intn(classes)
		err := rbf.createSpeech(pattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = rbf.createSpeech(pattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				return fmt.Errorf("createSpeech error: %v", err.Error())
			}
		}

		samp.desired = pattern

		// create spectrogram
		err = rbf.createSpectrogram(&samp)
		if err != nil {
			fmt.Printf("createSpectrogram error: %v\n", err.Error())
			return fmt.Errorf("createSpectrogram error: %v", err.Error())
		}

		// Forward Propagation
		err = rbf.propagateForward(&samp)
		if err != nil {
			return fmt.Errorf("forward propagation error: %s", err.Error())
		}

		// At the end of each epoch, classify the result
		err = rbf.determineClass(&samp)
		if err != nil {
			return fmt.Errorf("determineClass error: %s", err.Error())
		}
	}

	rbf.plot.TestResults = make([]Results, classes)

	totalCount := 0
	totalCorrect := 0
	classCount := 0
	// tabulate TestResults by converting numbers to string in Results
	for i := range rbf.plot.TestResults {
		classCount = rbf.statistics.classCount[i]
		totalCount += classCount
		totalCorrect += rbf.statistics.correct[i]
		if classCount > 0 {
			rbf.plot.TestResults[i] = Results{
				Class:   strconv.Itoa(i),
				Count:   strconv.Itoa(classCount),
				Correct: strconv.Itoa(rbf.statistics.correct[i] * 100 / classCount),
			}
		} else {
			rbf.plot.TestResults[i] = Results{
				Class:   strconv.Itoa(i),
				Count:   strconv.Itoa(classCount),
				Correct: "0",
			}
		}
	}
	rbf.plot.TotalCount = strconv.Itoa(totalCount)
	rbf.plot.TotalCorrect = strconv.Itoa(totalCorrect * 100 / totalCount)
	rbf.plot.LearningRate = strconv.FormatFloat(rbf.learningRate, 'f', 4, 64)
	rbf.plot.Epochs = strconv.Itoa(rbf.epochs)

	rbf.plot.Status = "Testing results completed."

	rbf.plot.LearningRate = strconv.FormatFloat(rbf.learningRate, 'f', 5, 64)
	rbf.plot.Momentum = strconv.FormatFloat(rbf.momentum, 'f', 5, 64)
	rbf.plot.HiddenLayers = strconv.Itoa(rbf.hiddenLayers)
	rbf.plot.LayerDepth = strconv.Itoa(rbf.layerDepth)
	rbf.plot.Epochs = strconv.Itoa(rbf.epochs)
	rbf.plot.FFTSize = strconv.Itoa(rbf.fftSize)
	rbf.plot.FFTWindow = rbf.fftWindow
	rbf.plot.Classes = strconv.Itoa(classes)
	rbf.plot.DelPitch = strconv.Itoa(rbf.delPitch)
	rbf.plot.DelDuration = strconv.Itoa(rbf.delDuration)
	rbf.plot.PercentVoiced = strconv.Itoa(rbf.percentVoiced)
	rbf.plot.DelAmpl = strconv.FormatFloat(rbf.delAmpl, 'f', -1, 64)

	return nil
}

// newTestingRBF constructs an RBF from the saved rbf weights and parameters
func newTestingRBF(plot *PlotT) (*RBF, error) {

	// Read in weights from csv file, ordered by layers, and RBF parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	items := strings.Split(line, ",")
	if len(items) != 11 {
		fmt.Printf("Testing parameters missing, should be 9, is %d\n", len(items))
		return nil, fmt.Errorf("testing parameters missing, run Train first")
	}

	epochs, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[0], err)
		return nil, err
	}

	hiddenLayers, err := strconv.Atoi(items[1])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[1], err)
		return nil, err
	}
	layerDepth, err := strconv.Atoi(items[2])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[2], err)
		return nil, err
	}

	learningRate, err := strconv.ParseFloat(items[3], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[4], err)
		return nil, err
	}

	momentum, err := strconv.ParseFloat(items[4], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[5], err)
		return nil, err
	}

	fftSize, err := strconv.Atoi(items[5])
	if err != nil {
		fmt.Printf("Conversion to int of 'fftSize' error: %v\n", err)
		return nil, err
	}

	fftWindow := items[6]

	delPitch, err := strconv.Atoi(items[7])
	if err != nil {
		fmt.Printf("Conversion to int of 'delPitch' error: %v\n", err)
		return nil, err
	}

	delDuration, err := strconv.Atoi(items[8])
	if err != nil {
		fmt.Printf("Conversion to int of delDuration' error: %v\n", err)
		return nil, err
	}

	percentVoiced, err := strconv.Atoi(items[9])
	if err != nil {
		fmt.Printf("Conversion to int of percentVoiced error: %v\n", err)
		return nil, err
	}

	delAmpl, err := strconv.ParseFloat(items[10], 64)
	if err != nil {
		fmt.Printf("Conversion to float of delAmpl error: %v\n", err)
		return nil, err
	}

	// construct the RBF
	rbf := RBF{
		epochs:       epochs,
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: 0.0,
			ymax: 100.0,
			xmin: 0,
			xmax: float64(classes - 1)},
		learningRate:  learningRate,
		momentum:      momentum,
		words:         make([]string, 0),
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		delDuration:   delDuration,
		delPitch:      delPitch,
		delAmpl:       delAmpl,
		percentVoiced: percentVoiced,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
		cluster:       make([]Kmeans, layerDepth),
	}

	// make SpeechFrames for the speech patterns
	nsamples := rbf.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

	// construct link that holds the weights and weight deltas
	rbf.link = make([][]Link, hiddenLayers+1)

	// input layer nodes are largest PSD bins for each STFT
	ilnodes := classes
	nwgts := ilnodes * layerDepth

	layer := 0
	rbf.link[layer] = make([]Link, nwgts)
	for i := 0; i < nwgts; i++ {
		scanner.Scan()
		line := scanner.Text()
		wgt, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Printf("ParseFloat input layer error: %v\n", err.Error())
			continue
		}
		rbf.link[0][i] = Link{wgt: wgt, wgtDelta: 0}
	}

	layer++
	// outer layer nodes
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer weights
	nwgts = layerDepth * olnodes
	rbf.link[layer] = make([]Link, nwgts)

	// Continue with output layer, one weight per line
	for i := 0; i < nwgts; i++ {
		scanner.Scan()
		line = scanner.Text()
		wt, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Printf("ParseFloat output layer error: %v", err.Error())
			continue
		}
		rbf.link[layer][i] = Link{wgt: wt, wgtDelta: 0}
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	// construct nodes
	rbf.node = make([][]Node, hiddenLayers+2)

	// input layer
	rbf.node[0] = make([]Node, ilnodes)

	// output layer
	rbf.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layer
	for i := 1; i <= hiddenLayers; i++ {
		rbf.node[i] = make([]Node, layerDepth)
	}

	// *********************************************************
	// construct desired from classes
	rbf.desired = make([]float64, olnodes)

	// percent correct classification of speech patterns
	rbf.mse = make([]float64, classes)

	// synthetic speech for creating speech with synthesize
	rbf.synSpeech = make([]float64, nsamples)

	// K-means cluster data for centroids
	for i := range layerDepth {
		rbf.cluster[i].mean = make([]float64, classes)
	}

	// make SpeechFrames for the speech patterns
	// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
	rbf.speechPat = make([][]SpeechFrame, classes)
	for i := range rbf.speechPat {
		rbf.speechPat[i] = make([]SpeechFrame, nframes)
	}

	files, err := os.ReadDir(dataDir)
	if err != nil {
		fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
		return nil, fmt.Errorf("ReadDir %s error: %v", dataDir, err)
	}
	if len(files) == 0 {
		fmt.Printf("No synthetic speech files in %s\n", dataDir)
		return nil, fmt.Errorf("no synthetic speech files in %s", dataDir)
	} else {
		pattern := 0
		// Retrieve the speech files
		for _, dirEntry := range files {
			name := dirEntry.Name()
			if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
				fspeech, err := os.Open(filepath.Join(dataDir, name))
				if err != nil {
					fmt.Printf("Open %s error: %v\n", name, err)
					return nil, fmt.Errorf("open error: %s", err.Error())
				}
				scanner := bufio.NewScanner(fspeech)
				frame := 0
				for scanner.Scan() {
					line := scanner.Text()
					items := strings.Split(line, ",")
					nfreqs := len(items) / 2
					rbf.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
					rbf.speechPat[pattern][frame].amps = make([]float64, nfreqs)
					for i := 0; i < nfreqs; i++ {
						freq, err := strconv.ParseFloat(items[i], 64)
						if err != nil {
							return nil, fmt.Errorf("freq %d conversion error: %v", i, err.Error())
						}
						ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
						if err != nil {
							return nil, fmt.Errorf("ampl %d conversion error: %v", i, err.Error())
						}
						rbf.speechPat[pattern][frame].amps[i] = ampl
						rbf.speechPat[pattern][frame].freqs[i] = freq
					}
					frame++
				}
				fspeech.Close()
				if err = scanner.Err(); err != nil {
					fmt.Printf("speech file scanner error: %s", err.Error())
					return nil, fmt.Errorf("speech file scanner error: %s", err.Error())
				}
				pattern++
			}
			if err = scanner.Err(); err != nil {
				return nil, fmt.Errorf("speech parameter scanner error: %s", err.Error())
			}
		}

		// retrieve K-means cluster data consisting of RBF prototypes
		if err := rbf.getKmeansCluster(&Endpoints{}); err != nil {
			fmt.Printf("getKmeansCluster error: %v\n", err)
			return nil, fmt.Errorf("getKmeansCluster error: %v", err)
		}
	}
	return &rbf, nil
}

// handleTestingRBF performs pattern classification of the test data
func handleTestingRBF(w http.ResponseWriter, r *http.Request) {
	// create synthetic speech
	// loop over the synthetic speech and generate the spectrogram
	// propagate forward and classify the output
	// fill the grid with the percent correct

	var (
		plot PlotT
		rbf  *RBF
		err  error
	)

	// Construct RBF instance containing RBF state
	rbf, err = newTestingRBF(&plot)
	if err != nil {
		fmt.Printf("newTestingRBF() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingRBF() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingRBF.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// At end of all examples display TestingResults
	// Convert classification numbers to string in Results
	err = rbf.runTestingEpochs()
	if err != nil {
		fmt.Printf("runTestingEpochs() error: %v\n", err)
		plot.Status = fmt.Sprintf("runTestingEpochs() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingRBF.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Put the percent correct in data
	for pat := range rbf.mse {
		rbf.mse[pat] = float64(rbf.statistics.correct[pat]) / float64(rbf.statistics.classCount[pat]) * 100.0
	}
	// Put data in PlotT
	err = rbf.gridFillInterp()
	if err != nil {
		fmt.Printf("gridFillInterp() error: %v\n", err)
		plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingRBF.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// insert x-labels and y-labels in PlotT
	rbf.insertLabels()

	// Execute data on HTML template
	if err = tmplTestingRBF.Execute(w, rbf.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// findEndpoints finds the minimum and maximum data values
func (ep *Endpoints) findEndpoints(input []float64) {
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	for _, y := range input {

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
}

// processTimeDomain plots the time domain data of the synthetic speech
func (rbf *RBF) processTimeDomain(pattern int) error {

	var (
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	rbf.plot.Grid = make([]string, rows*cols)
	rbf.plot.Xlabel = make([]string, xlabels)
	rbf.plot.Ylabel = make([]string, ylabels)

	rbf.nsamples = len(rbf.synSpeech)

	endpoints.findEndpoints(rbf.synSpeech)
	// time starts at 0 and ends at #samples*sampling period
	endpoints.xmin = 0.0
	// #samples*sampling period, sampling period = 1/sampleRate
	endpoints.xmax = float64(rbf.nsamples) / float64(sampleRate)

	// EP means endpoints
	lenEPx := endpoints.xmax - endpoints.xmin
	lenEPy := endpoints.ymax - endpoints.ymin
	prevTime := 0.0
	prevAmpl := rbf.synSpeech[0]

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// This previous cell location (row,col) is on the line (visible)
	row := int((endpoints.ymax-rbf.synSpeech[0])*yscale + .5)
	col := int((0.0-endpoints.xmin)*xscale + .5)
	rbf.plot.Grid[row*cols+col] = "online"

	// Store the amplitude in the plot Grid
	for n := 1; n < rbf.nsamples; n++ {
		// Current time
		currTime := float64(n) / float64(sampleRate)

		// This current cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-rbf.synSpeech[n])*yscale + .5)
		col := int((currTime-endpoints.xmin)*xscale + .5)
		rbf.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point;
		// draw a straight line between points.
		lenEdgeTime := math.Abs((currTime - prevTime))
		lenEdgeAmpl := math.Abs(rbf.synSpeech[n] - prevAmpl)
		ncellsTime := int(float64(cols) * lenEdgeTime / lenEPx) // number of points to interpolate in x-dim
		ncellsAmpl := int(float64(rows) * lenEdgeAmpl / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsAmpl, ncellsTime)

		stepTime := float64(currTime-prevTime) / float64(ncells)
		stepAmpl := float64(rbf.synSpeech[n]-prevAmpl) / float64(ncells)

		// loop to draw the points
		interpTime := prevTime
		interpAmpl := prevAmpl
		for i := 0; i < ncells; i++ {
			row := int((endpoints.ymax-interpAmpl)*yscale + .5)
			col := int((interpTime-endpoints.xmin)*xscale + .5)
			// This cell location (row,col) is on the line (visible)
			rbf.plot.Grid[row*cols+col] = "online"
			interpTime += stepTime
			interpAmpl += stepAmpl
		}

		// Update the previous point with the current point
		prevTime = currTime
		prevAmpl = rbf.synSpeech[n]

	}

	// Set plot status if no errors
	if len(rbf.plot.Status) == 0 {
		rbf.plot.Status = fmt.Sprintf("speech pattern %d plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			pattern, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range rbf.plot.Xlabel {
		rbf.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range rbf.plot.Ylabel {
		rbf.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// processSpectrogram creates a spectrogram of the speech waveform
func (rbf *RBF) processSpectrogram(pattern int, fftWindow string, fftSize int) error {

	// get speech samples from rbf.synSpeech
	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
	)
	fftSize2 := fftSize / 2

	rbf.plot.Grid = make([]string, rows*cols)
	rbf.plot.Xlabel = make([]string, xlabels)
	rbf.plot.Ylabel = make([]string, ylabels)

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, fftSize/2)

	rbf.nsamples = len(rbf.synSpeech)
	// x-axis is time or sample, y-axis is frequency
	endpoints.xmin = 0.0
	endpoints.xmax = float64(rbf.nsamples)
	endpoints.ymin = 0.0
	endpoints.ymax = float64(fftSize2) // equivalent to Nyquist critical frequency

	// Calculate scale factors to convert physical units to screen units
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// number of cells to interpolate in time and frequency, stepping by fftSize/2 in time for ncellst
	// round up so the cells in the plot grid are connected
	ncellst := int((math.Ceil(float64(cols) * float64(fftSize2) / float64(rbf.nsamples))))
	ncellsf := int(math.Ceil(float64(rows) / float64((fftSize2))))

	stepTime := float64((fftSize) / ncellst)
	stepFreq := 1.0 / float64(ncellsf)

	// for loop over samples, increment by fftSize/2, calculatePSD on the batch
	// Overlap by 50% due to non-rectangular window to avoid Gibbs phenomenon
	for smpl := 0; smpl < rbf.nsamples; smpl += fftSize2 {
		// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
		end := smpl + fftSize
		if end > rbf.nsamples {
			end = rbf.nsamples
		}
		_, psdMax, err := rbf.calculatePSD(rbf.synSpeech[smpl:end], PSD, fftWindow, fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}

		// for loop over the frequency bins in the PSD
		for bin := 0; bin < fftSize2; bin++ {
			// find the grayscale color based on bin power
			// largest power is black, smallest power is white
			// shades of gray in-between black and white
			var gs string
			r := PSD[bin] / psdMax
			if r < .10 {
				gs = rbf.grayscale[4]
			} else if r < .25 {
				gs = rbf.grayscale[3]
			} else if r < .50 {
				gs = rbf.grayscale[2]
			} else if r < .80 {
				gs = rbf.grayscale[1]
			} else {
				gs = rbf.grayscale[0]
			}

			// interpolate in time
			interpTime := float64(smpl)
			for nct := 0; nct < ncellst; nct++ {
				col := int((interpTime-endpoints.xmin)*xscale + .5)
				if col >= cols {
					col = cols - 1
				}
				// interpolate in frequency
				interpFreq := float64(bin)
				for ncf := 0; ncf < ncellsf; ncf++ {
					row := int((endpoints.ymax-interpFreq)*yscale + .5)
					if row < 0 {
						row = 0
					}
					// Store the color in the plot Grid
					rbf.plot.Grid[row*cols+col] = gs
					interpFreq += stepFreq
				}
				interpTime += stepTime
			}
		}
	}

	// Set plot status if no errors
	if len(rbf.plot.Status) == 0 {
		rbf.plot.Status = fmt.Sprintf("spectrogram of pattern %d plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			pattern, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / ((xlabels - 1) * sampleRate)
	x := endpoints.xmin / sampleRate
	// First label is empty for alignment purposes
	for i := range rbf.plot.Xlabel {
		rbf.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Apply the  sampling rate in Hz to the y-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.ymax

	// Construct y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	// First label is empty for alignment purposes
	for i := range rbf.plot.Ylabel {
		rbf.plot.Ylabel[i] = fmt.Sprintf("%.0f", y*sf)
		y += incr
	}

	return nil
}

// K-Means Within Class Squared Sum (WCSS) versus K
func (rbf *RBF) processKmeansCluster() error {
	var (
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	// retrieve K-means cluster data consisting of RBF prototypes
	// get the min and max of wcss and save in endpoints
	if err := rbf.getKmeansCluster(&endpoints); err != nil {
		fmt.Printf("getKmeansCluster error: %v\n", err)
		return fmt.Errorf("getKmeansCluster error: %v", err)
	}

	rbf.plot.Grid = make([]string, rows*cols)
	rbf.plot.Xlabel = make([]string, xlabels)
	rbf.plot.Ylabel = make([]string, ylabels)

	rbf.nsamples = len(rbf.cluster)

	// time starts at 0 and ends at number of centroids = classes
	endpoints.xmin = 1.0
	// max value of K in K-means cluster
	endpoints.xmax = float64(rbf.layerDepth)
	// endpoints for ymin and ymax found in call to getKmeansCluster above

	// EP means endpoints
	lenEPx := endpoints.xmax - endpoints.xmin
	lenEPy := endpoints.ymax - endpoints.ymin
	prevTime := 1.0
	prevAmpl := rbf.cluster[0].wcss

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// Current time
	currTime := 1.0
	// This previous cell location (row,col) is on the line (visible)
	row := int((endpoints.ymax-rbf.cluster[int(currTime-1.0)].wcss)*yscale + .5)
	col := int((currTime-endpoints.xmin)*xscale + .5)
	rbf.plot.Grid[row*cols+col] = "online"

	// Store the amplitude in the plot Grid
	for n := 2; n <= rbf.nsamples; n++ {
		currTime = float64(n)

		// This current cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-rbf.cluster[n-1].wcss)*yscale + .5)
		col := int((currTime-endpoints.xmin)*xscale + .5)
		rbf.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point;
		// draw a straight line between points.
		lenEdgeTime := math.Abs((currTime - prevTime))
		lenEdgeAmpl := math.Abs(rbf.cluster[n-1].wcss - prevAmpl)
		ncellsTime := int(float64(cols) * lenEdgeTime / lenEPx) // number of points to interpolate in x-dim
		ncellsAmpl := int(float64(rows) * lenEdgeAmpl / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsAmpl, ncellsTime)

		stepTime := float64(currTime-prevTime) / float64(ncells)
		stepAmpl := float64(rbf.cluster[n-1].wcss-prevAmpl) / float64(ncells)

		// loop to draw the points
		interpTime := prevTime
		interpAmpl := prevAmpl
		for range ncells {
			row := int((endpoints.ymax-interpAmpl)*yscale + .5)
			col := int((interpTime-endpoints.xmin)*xscale + .5)
			// This cell location (row,col) is on the line (visible)
			rbf.plot.Grid[row*cols+col] = "online"
			interpTime += stepTime
			interpAmpl += stepAmpl
		}

		// Update the previous point with the current point
		prevTime = currTime
		prevAmpl = rbf.cluster[n-1].wcss

	}

	// Set plot status if no errors
	if len(rbf.plot.Status) == 0 {
		rbf.plot.Status = fmt.Sprintf("K-means Cluster WCSS plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range rbf.plot.Xlabel {
		rbf.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range rbf.plot.Ylabel {
		rbf.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// newDisplayRBF creates a RBF instance for displaying speech in time or spectrogram
func newDisplayRBF(r *http.Request, plot *PlotT) (*RBF, error) {

	// Get from form percent voiced, pitch variation, duration variation
	txt := r.FormValue("delpitch")
	delPitch, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delPitch int conversion error: %v\n", err)
		return nil, fmt.Errorf("delPitch conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("delduration")
	delDuration, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delDuration int conversion error: %v\n", err)
		return nil, fmt.Errorf("delDuration conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("percentvoiced")
	percentVoiced, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("percentVoiced int conversion error: %v\n", err)
		return nil, fmt.Errorf("percentVoiced conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("delampl")
	delAmpl, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("delta Amplitude conversion error: %v\n", err)
		return nil, err
	}

	// Read in RBF parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	items := strings.Split(line, ",")
	if len(items) != 11 {
		fmt.Printf("Display parameters missing, should be 11, is %d\n", len(items))
		return nil, fmt.Errorf("display parameters missing, should be 11, is %d", len(items))
	}

	layerDepth, err := strconv.Atoi(items[2])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[2], err)
		return nil, err
	}

	fftSize, err := strconv.Atoi(items[5])
	if err != nil {
		fmt.Printf("Conversion to int of 'fftSize' error: %v\n", err)
		return nil, err
	}

	fftWindow := items[6]

	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s\n", err.Error())
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	rbf := RBF{
		plot:          plot,
		delPitch:      delPitch,
		delDuration:   delDuration,
		percentVoiced: percentVoiced,
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
		delAmpl:       delAmpl,
		cluster:       make([]Kmeans, layerDepth),
		layerDepth:    layerDepth,
	}

	// Determine if time, K-means cluster, or spectrogram domain plot
	rbf.domain = r.FormValue("domain")
	switch rbf.domain {
	case "spectrogram":
		plot.Domain = "Spectrogram (Hz/sec)"
	case "kmeanscluster":
		plot.Domain = "K-Means Cluster (WCSS/K)"
	default:
		plot.Domain = "Time Domain (sec)"
	}

	// synthetic speech for creating speech with synthesize
	rbf.synSpeech = make([]float64, rbf.fftSize*nffts)

	// K-means cluster data for centroids
	for i := range layerDepth {
		rbf.cluster[i].mean = make([]float64, classes)
	}

	return &rbf, nil
}

// handleDisplayRBF displays the selected speech pattern (time or spectrogram) and plays the audio
func handleDisplayRBF(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		rbf  *RBF
	)

	// Get the speech pattern to display
	txt := r.FormValue("speechpattern")
	// Need speech pattern to continue
	if len(txt) > 0 {
		speechPattern, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Speech Pattern int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Speech Pattern conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Construct RBF instance containing RBF state
		rbf, err = newDisplayRBF(r, &plot)
		if err != nil {
			fmt.Printf("newDisplayRBF() error: %v\n", err)
			plot.Status = fmt.Sprintf("newDisplayRBF() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// make SpeechFrames for the speech patterns
		nsamples := rbf.fftSize * nffts
		// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
		nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

		// create new synthetic speech patterns
		newPattern := r.FormValue("newpattern")
		if len(newPattern) > 0 {
			if err = rbf.createPatterns(); err != nil {
				fmt.Printf("createPatterns() error: %v\n", err)
				plot.Status = fmt.Sprintf("createPatterns() error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			// Read the synthetic speech patterns
		} else {
			files, err := os.ReadDir(dataDir)
			if err != nil {
				fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
				plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			if len(files) == 0 {
				fmt.Printf("No synthetic speech files in %s\n", dataDir)
				plot.Status = fmt.Sprintf("No synthetic speech files in %s, create new patterns", dataDir)
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				rbf.speechPat = make([][]SpeechFrame, classes)
				for i := range rbf.speechPat {
					rbf.speechPat[i] = make([]SpeechFrame, nframes)
				}
				pattern := 0
				// Retrieve the speech files
				for _, dirEntry := range files {
					name := dirEntry.Name()
					if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
						fspeech, err := os.Open(filepath.Join(dataDir, name))
						if err != nil {
							fmt.Printf("Open %s error: %v\n", name, err)
							plot.Status = fmt.Sprintf("Open %s error: %v", name, err.Error())
							// Write to HTTP using template and grid
							if err := tmplTrainingRBF.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						scanner := bufio.NewScanner(fspeech)
						frame := 0
						for scanner.Scan() {
							line := scanner.Text()
							items := strings.Split(line, ",")
							nfreqs := len(items) / 2
							rbf.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
							rbf.speechPat[pattern][frame].amps = make([]float64, nfreqs)
							for i := 0; i < nfreqs; i++ {
								freq, err := strconv.ParseFloat(items[i], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("freq %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTrainingRBF.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("ampl %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTrainingRBF.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								rbf.speechPat[pattern][frame].amps[i] = ampl
								rbf.speechPat[pattern][frame].freqs[i] = freq
							}
							frame++
						}
						fspeech.Close()
						if err = scanner.Err(); err != nil {
							fmt.Printf("speech file scanner error: %s", err.Error())
							// Write to HTTP using template and grid
							if err := tmplTrainingRBF.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						pattern++
					}
				}
			}
		}

		// generate speech, spectrogram if desired, create wav file,
		// using speech pattern, percent voiced, deltas duration, amplitude, and pitch

		err = rbf.createSpeech(speechPattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = rbf.createSpeech(speechPattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				plot.Status = fmt.Sprintf("createSpeech error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}

		switch rbf.domain {
		case "spectrogram":
			rbf.grayscale = make(map[int]string)
			for i := range ncolors {
				rbf.grayscale[i] = fmt.Sprintf("gs%d", i)
			}

			err := rbf.processSpectrogram(speechPattern, rbf.fftWindow, rbf.fftSize)
			if err != nil {
				fmt.Printf("proessSpectrogram error: %v\n", err)
				plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = fmt.Sprintf("Spectrogram of pattern %d plotted.", speechPattern)
		case "kmeanscluster":
			err := rbf.processKmeansCluster()
			if err != nil {
				fmt.Printf("processKmeansCluster error: %v\n", err)
				plot.Status = fmt.Sprintf("processKmeansCluster error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = "K-Means Within Class Squared Sum (WCSS) plotted."
		default:
			err := rbf.processTimeDomain(speechPattern)
			if err != nil {
				fmt.Printf("processTimeDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplayRBF.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = fmt.Sprintf("Time Domain of pattern %d plotted.", speechPattern)
		}

		// Create the wav file from the synthetic speech
		outF, err := os.Create(path.Join(dataDir, synSpeech))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", synSpeech, err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", synSpeech, err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer outF.Close()
		// create wav.Encoder
		enc := wav.NewEncoder(outF, sampleRate, bitDepth, 1, 1)

		// create audio.FloatBuffer
		float64Buf := &audio.FloatBuffer{Data: rbf.synSpeech, Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate}}

		// create IntBuffer from FloatBuffer and pass to Encoder.Write()
		if err := enc.Write(float64Buf.AsIntBuffer()); err != nil {
			fmt.Printf("wav encoder write error: %v\n", err)
			plot.Status = fmt.Sprintf("wav encoder write error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// close the encoder
		if err := enc.Close(); err != nil {
			fmt.Printf("wav encoder close error: %v\n", err)
			plot.Status = fmt.Sprintf("wav encoder close error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplayRBF.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Play the audio wav if fmedia is available in the PATH environment variable
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			cmd := exec.Command(fmedia, filepath.Join(dataDir, synSpeech))
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}

		// set the speech parameters
		rbf.plot.DelDuration = strconv.Itoa(rbf.delDuration)
		rbf.plot.DelPitch = strconv.Itoa(rbf.delPitch)
		rbf.plot.DelAmpl = strconv.FormatFloat(rbf.delAmpl, 'f', -1, 64)
		rbf.plot.PercentVoiced = strconv.Itoa(rbf.percentVoiced)
		rbf.plot.SpeechPattern = strconv.Itoa(speechPattern)

		// Execute plot on display HTML template
		if err = tmplDisplayRBF.Execute(w, rbf.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Display speech parameters:  pattern, percent voiced, pitch delta, duration delta."
		// Write to HTTP using template and grid
		if err := tmplDisplayRBF.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the Radial Basis Function Neural Network

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingRBF, handleTrainingRBF)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingRBF, handleTestingRBF)
	// Create HTTP handler for spectrogram generation
	http.HandleFunc(patternDisplayRBF, handleDisplayRBF)
	fmt.Printf("Speech RBF Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
