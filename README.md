# EEC 201 FINAL PROJECT
Speaker recognition

## Introduction

Speaker recognition is a popular task in the field of Speech processing. This task is designed to match the audio to different speakers to verify the identity of each speaker. In current research, Mel-frequency cepstrum technology has demonstrated great performance since it is close to the human auditory system. In addition, vector quantization is a data compression technology that makes signal processing more efficient than original data. Hence, In this project, we implement a speaker recognition algorithm based on the Mel-frequency cepstrum coefficients (MFCCs) processor and Vector Quantization method. The recognition accuracy of the algorithm has achieved satisfactory results in different tests in this project.

## A. Speech Data

In this project, the algorithm is implemented on three datasets: **non-student speech**, **2024 Student Audio Recording**, and **2025 Student Audio Recording**. The **Non-student speech** dataset consists of 19 audio recordings, all of which contain the word "zero." The training set contains 11 samples, and the remaining 11 samples constitute the testing set.

The **2024 Student Audio Recording** dataset consists of 76 audio recordings, including 38 recordings of "zero" and 38 recordings of "twelve." Half of "zero" forms the training set, and the remaining half of "zero" forms the testing set. The same split applies to the "twelve" recordings.

Similarly, the **2025 Student Audio Recording** dataset consists of 92 audio recordings, including 46 recordings of "five" and 46 recordings of "eleven." Half of "five" forms the training set, and the remaining half forms the testing set. The same split applies to the "eleven" recordings.

### Test 1:

In this test, we first played the training samples from the **non-student speech** dataset. Then, we played the testing samples from the same dataset in random order for manual recognition and identity matching. Then, we recorded our manual recognition results as a benchmark accuracy rate to compare with the algorithm results in the following tests to evaluate the algorithm performance.

For the **non-student speech** dataset, the accuracy rate obtained through our human auditory is **62.5%**.

## B. Speech Preprocessing

### Test 2:

The sampling rate: 12500

The milliseconds: 20.48

### Use STFT to generate periodogram

#### Example for the first speaker

Frame size: N=128

![stft_128](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/stft_128.png)

Frame size: N=256

![stft_256](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/stft_256.png)

Frame size: N=512

![stft_512](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/stft_512.png)

### Test 3:

Example for the first speaker

![Mel-spaced filter bank responses](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_3_1.png)
![test 3 2](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_3_2.png)
![test 3 3](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_3_3.png)

### Test 5:

Example for the first three speakers

![MFCC](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_5.png)

### Test 6:

Example for the first three speakers

![VQ codewords](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_6.png)

### Test 7:
Accuracy rate = 100%

### Test 8:
Accuracy rate = 87.5%

### Test 9_Five:
Accuracy rate = 95.65%

### Test 9_Eleven:
Accuracy rate = 95.65%

### Test 10:
Accuracy rate = 97.83%
