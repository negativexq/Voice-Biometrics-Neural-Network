# Voice Biometrics Neural Network

A neural network-based **speaker identification** project that classifies voice recordings among 30 known speakers using handcrafted acoustic features extracted from audio files.

## Overview

This project implements a closed-set speaker identification system.

Given a `.flac` audio recording, the model predicts which one of the 30 registered speakers produced the recording.

The project combines traditional audio feature extraction with a fully connected neural network:

```text
Audio Recording
      ↓
Acoustic Feature Extraction
      ↓
193-Dimensional Feature Vector
      ↓
Feature Standardization
      ↓
Dense Neural Network
      ↓
Speaker Prediction
```

> This project performs speaker identification, not speaker verification. It predicts one speaker from a predefined set of known speakers rather than comparing two recordings to determine whether they belong to the same person.

## Key Features

* Speaker classification across 30 known speakers
* Audio processing with Librosa
* Support for `.flac` audio recordings
* Speaker labels extracted from file names
* Acoustic feature extraction
* Feature standardization
* Dense neural network classification
* Separate training, validation and test datasets
* Achieved 96.7% accuracy on the test set

## Dataset Structure

The dataset is divided into three folders:

```text
Voice-Biometrics-Neural-Network/
│
├── 30_speakers_train/
├── 30_speakers_val/
├── 30_speakers_test/
├── features_train.npy
├── features_val.npy
├── features_test.npy
└── Voice-Classification-Neural-Network.ipynb
```

The dataset contains 30 different speakers.

| Dataset    | Audio Samples |
| ---------- | ------------: |
| Training   |           120 |
| Validation |            90 |
| Test       |            90 |

Speaker IDs are extracted from the beginning of each audio file name.

Example:

```text
7278-91083-0000.flac
^^^^
Speaker ID
```

## Audio Feature Extraction

Each audio recording is loaded using Librosa with a default sampling rate of 22,050 Hz.

The following acoustic features are extracted:

### MFCC

40 Mel-Frequency Cepstral Coefficients are calculated to represent the spectral characteristics of the human voice.

### Chroma STFT

Chroma features describe the distribution of audio energy across the 12 pitch classes.

### Mel Spectrogram

The Mel-scaled spectrogram represents the frequency content of the audio using a scale closer to human auditory perception.

### Spectral Contrast

Spectral contrast measures the difference between peaks and valleys in different frequency bands.

### Tonnetz

Tonal centroid features represent harmonic and tonal relationships in the audio signal.

The extracted features are averaged across time and concatenated into a single vector:

```text
MFCC
 + Chroma
 + Mel Spectrogram
 + Spectral Contrast
 + Tonnetz
-----------------------------
193-Dimensional Feature Vector
```

## Data Preprocessing

Before training, the extracted feature vectors are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

Speaker labels are encoded using `LabelEncoder` and converted into categorical values using one-hot encoding.

## Neural Network Architecture

The model is implemented using Keras and follows a fully connected architecture.

```text
Input Layer
193 features
      ↓
Dense Layer
193 neurons, ReLU
      ↓
Dropout
0.10
      ↓
Dense Layer
128 neurons, ReLU
      ↓
Dropout
0.25
      ↓
Dense Layer
128 neurons, ReLU
      ↓
Dropout
0.50
      ↓
Output Layer
30 neurons, Softmax
```

Model definition:

```python
model = Sequential()

model.add(Dense(193, input_shape=(193,), activation="relu"))
model.add(Dropout(0.10))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.50))

model.add(Dense(30, activation="softmax"))
```

The model is compiled using:

* Adam optimizer
* Categorical cross-entropy loss
* Accuracy metric
* Early stopping

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

## Results

The model achieved the following performance on the test dataset:

```text
Test samples: 90
Correct predictions: 87
Incorrect predictions: 3
Test accuracy: 96.7%
```

Example predictions:

| Actual Speaker | Predicted Speaker | Result    |
| -------------- | ----------------- | --------- |
| 7312           | 7312              | Correct   |
| 8014           | 8014              | Correct   |
| 8088           | 8088              | Correct   |
| 7505           | 7402              | Incorrect |
| 7367           | 8419              | Incorrect |

## Installation

Clone the repository:

```bash
git clone https://github.com/negativexq/Voice-Biometrics-Neural-Network.git
cd Voice-Biometrics-Neural-Network
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib librosa scikit-learn tensorflow keras jupyter
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Voice-Classification-Neural-Network.ipynb
```

Run the notebook cells sequentially to:

1. Load the training, validation and test audio files.
2. Extract speaker IDs from file names.
3. Extract acoustic features.
4. Standardize the feature vectors.
5. Train the neural network.
6. Generate test predictions.
7. Evaluate classification accuracy.

## Technologies

* Python
* Jupyter Notebook
* TensorFlow
* Keras
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## Limitations

This project uses a closed-set speaker identification approach.

The current model:

* Can only classify speakers included in the training dataset
* Does not verify whether two recordings belong to the same person
* Does not generate speaker embeddings
* Does not support unknown-speaker rejection
* Uses averaged handcrafted features instead of raw audio sequences
* Does not include noise reduction or audio augmentation
* Has been evaluated on a relatively small dataset

Therefore, the reported accuracy should be interpreted within the scope of the provided 30-speaker dataset.

## Future Improvements

Possible improvements include:

* Refactoring notebook code into reusable Python modules
* Adding a `requirements.txt` file
* Adding reproducible train and inference scripts
* Applying audio augmentation
* Supporting background noise and different recording environments
* Adding confusion matrix and class-level metrics
* Implementing speaker embeddings
* Implementing Siamese or Triplet networks
* Adding cosine similarity-based speaker verification
* Evaluating Equal Error Rate, FAR and FRR
* Using architectures such as CNN, LSTM or ECAPA-TDNN
* Creating a FastAPI inference service
* Adding Docker support
* Adding experiment tracking with MLflow

## Speaker Identification vs Speaker Verification

| Speaker Identification                 | Speaker Verification                             |
| -------------------------------------- | ------------------------------------------------ |
| Determines who is speaking             | Determines whether a claimed identity is correct |
| Selects one speaker from known classes | Compares recordings or speaker embeddings        |
| Multiclass classification              | Similarity or distance-based decision            |
| Used in this project                   | Not currently implemented                        |

## Disclaimer

This repository is an experimental machine learning project created for educational purposes.

It should not be used as a production biometric authentication system without further work on security, privacy, robustness, bias evaluation, spoofing prevention and unknown-speaker detection.

## Author

**Ömer Faruk Koç**

* GitHub: [negativexq](https://github.com/negativexq)
* LinkedIn: [omerfkoc](https://www.linkedin.com/in/omerfkoc/)
* Email: [omerfkoc98@gmail.com](mailto:omerfkoc98@gmail.com)
