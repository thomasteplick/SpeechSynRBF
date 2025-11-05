<h3>
Synthetic Speech Classification using a Radial Basis Function (RBF) Neural Network with K-means clustering
</h3>
<p>
This is a web application written in Go that make use of the html/template package to dynamically
create the web page. Start the web server at bin\rbfspeech.exe and connect to it from your web browser
at http://127.0.0.1:8080/speechRBFtrain. The program creates synthetic speech and generates spectrograms from them. 
Spectrograms are 3-dimensional plots of the time-frequency content of the audio. The spectral power at a 
particular frequency and time is shown as a grayscale color, with black having the greatest power and white 
having the least. Short-time Fourier transforms (STFT) are used in 32 ms time frames with no overlap. 
between FFTs. Time domain plots of the audio waveform can be displayed as well as the spectrogram. 
The user can generate new synthetic speech of 2.048 seconds (16,384 samples) by selecting the New Speech radio button.  
Upon selecting the <i>Submit</i> button, the user will hear the speech on their computer's audio device. 
To hear the synthetic speech requires the <b>fmedia</b> program to be in the PATH environmental variable.
A WAV file is created from the synthetic speach and submitted to the fmedia program to be played.</p>

<p>
The FFT size is 256. The sampling rate is 8,000 Hz which produces a Nyquist critical frequency of 4,000 Hz. 
The grayscale consists of five colors. As stated above, black RGB(0,0,0), has the greatest power in the 
Power Spectral Density (PSD) bin.  There are 256 bins in the 256-pt FFT and each bin is 8,000/256 = 31.25 Hz wide.
</p>

<p>
The spectrogram's PSD maximum bin for each 32ms frame is used as input to the Multilayer Perceptron Neural Network. 
That is, the frequency bin with the maximum power of each frame is submitted serially to the input layer of the RBF. 
Backpropagation is used to train the weights in the network. The Train page allows the speech and frequency 
domain parameters to be chosen. Upon sumitting the form in the HTML, the synthetic speech patterns are converted 
to spectrograms and submitted to the RBF. Next the Test page will generate variation of the synthetic speech 
patterns. In the <i>Display</i> page, you can plot the time domain or the spectrogram of one of the synthetic speech patterns.
</p>

<p>
The synthetic speech is generated with a sum of sinusoids (voiced) or gaussian noise (unvoiced) in
20-30 ms frames.  If voiced, the fundamental frequency is randomly chosen from between 100 and 3800 Hz. Each voiced
speech has 1-5 subfrequencies with a smaller amplitude than the fundamental.  The amplitudes are randomly
chosen and can be varied.  The duration of each frame can also be varied.  The variation of these parameters
will test the generalization capabilities of the Neural Network.  The testing phase varies the parameters
based upon the user input.  The percentage of correct classification is presented in graphical and tabular
forms upon completion of the testing.
</p>

<p>
There are two phases of operation:  the training phase and the testing phase.  
Epochs consising of a sequence of examples are used to train the Neural Network.  
Each example consists of a spectrogram (STFT) of synthetic speech and a desired class output.  The RBF
itself consists of an input layer of nodes for the input, one hidden layer containing nodes with
Gaussian radial basis functions, and an output layer of nodes that are fully connected by weighted links
to the hidden layer of Gaussian radial basis functions. The hidden-to-output layer of
weights are trained by back propagating the output layer errors forward to the
hidden layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layer.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.  The RBF Neural Network uses hyperbolic tangent tanh activation function
and mean-square error (MSE) Loss function in the output layer.  Mean-square is used to compute
the error in the ouput layer with an encoded binary vector as the target or desired output.
This is a classification problem and the encoded binary output with log2(classes) dimension is 
the desired or target.  The number of classes or synthetic speech patterns is 64; therefore the output layer is dimension six.
</p>

<p>
K-means clustering is used to find the radial basis function centroid (mean) and bandwidth.  This is an iterative procedure in which
the following two steps repeat until the fractional <i>Within-class Squared Sum</i> (WCSS) delta is less than a threshold or
max iteration is reached.
<ul type="disc">
<li>Place the data in the cluster whose centroid is closest in Euclidean distance</li>
<li>Recalculate the cluster centroids using the points assigned to that cluster</li>
</ul>
The bandwidth or variance is found for each cluster by finding the sample variance
for all the points within each cluster.  The centroids or means are initialized by setting
the first centroid to a random point in the data.  The remaining centroids are found using a maxmin
procedure.  The minimum Euclidean distance of the data points in a cluster from their means is found and the
maximum of these distances is chosen as the next centroid.  This is repeated until all the desired centroids
are found.  The number of RBFs or clusters determines the successful classification of the data.  This number
should be greater than the input layer dimension and less than the number of training samples.  For example, if
the input layer dimension is 64 and the number of training samples is 3,000, a hidden layer depth of 128 would be
a good choice to classify the synthetic speech samples.
</p>

<h4>Training Learning Curve, Mean-square Error vs Epoch, 128 clusters</h4>
<img width="1462" height="903" alt="image" src="https://github.com/user-attachments/assets/d72e7954-cff1-405c-8977-d28d59190930" />
<h4>Test Results, Percent correct versus pattern or class, 128 clusters</h4>
<img width="1459" height="992" alt="image" src="https://github.com/user-attachments/assets/6f64f869-eb51-4f51-a0fa-834b40d3a099" />
<img width="1458" height="668" alt="image" src="https://github.com/user-attachments/assets/a41ce620-3c3d-45ca-a398-ba5fe04c33ac" />
<h4>Display Time Domain, Speech Pattern 1, 128 clusters</h4>
<img width="1464" height="993" alt="image" src="https://github.com/user-attachments/assets/522272ec-12c0-41f0-8451-c683bfbb8ba6" />
<h4>Display Spectrogram, Speech Pattern 1, 128 clusters</h4>
<img width="1461" height="990" alt="image" src="https://github.com/user-attachments/assets/ad9c8bcb-3a54-472b-a679-f049d9abd18b" />
<h4>K-means Cluster, Within-class Summed Square (WCSS/K-means), 0-128 clusters</h4>
<img width="1460" height="990" alt="image" src="https://github.com/user-attachments/assets/3a0ef38b-9a26-4bac-b410-6da9755032d0" />
<h4>Training Learning Curve, Mean-square Error,  64 clusters</h4>
<img width="1462" height="993" alt="image" src="https://github.com/user-attachments/assets/537a284c-c5aa-4810-8319-e967386125a6" />
<h4>Test Results, Percent correct versus pattern or class,  64 clusters</h4>
<img width="1458" height="990" alt="image" src="https://github.com/user-attachments/assets/8292c15b-0144-40c7-87b4-5b47dadca440" />
<img width="1454" height="681" alt="image" src="https://github.com/user-attachments/assets/a423d71b-e856-4610-b256-297aafd31dba" />




