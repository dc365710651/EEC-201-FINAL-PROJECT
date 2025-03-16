# EEC 201 FINAL PROJECT
Speaker recognition

## Introduction

Speaker recognition is a popular task in the field of Speech processing. This task is designed to match the audio to different speakers to verify the identity of each speaker. In current research, Mel-frequency cepstrum technology has demonstrated great performance since it is close to the human auditory system. In addition, vector quantization is a data compression technology that makes signal processing more efficient than original data. Hence, In this project, we implement a speaker recognition algorithm based on the Mel-frequency cepstrum coefficients (MFCCs) processor and Vector Quantization method. The recognition accuracy of the algorithm has achieved satisfactory results in different tests in this project.

## A. Speech Data

In this project, the algorithm is implemented on three datasets: **non-student speech**, **2024 Student Audio Recording**, and **2025 Student Audio Recording**. The **Non-student speech** dataset consists of 19 audio recordings, all of which contain the word "zero." The training set contains 11 samples, and the remaining 11 samples constitute the testing set.

The **2024 Student Audio Recording** dataset consists of 76 audio recordings, including 38 recordings of "zero" and 38 recordings of "twelve." Half of "zero" forms the training set, and the remaining half of "zero" forms the testing set. The same split applies to the "twelve" recordings.

Similarly, the **2025 Student Audio Recording** dataset consists of 92 audio recordings, including 46 recordings of "five" and 46 recordings of "eleven." Half of "five" forms the training set, and the remaining half forms the testing set. The same split applies to the "eleven" recordings.

We develop this algorithm and train the recognition system based on **non-student speech** dataset from test 1 to test 8. And for test 9, we retrained the system on a mixed dataset combining **non-student speech** and **2024 Student Audio Recording** datasets. For test 10a and test 10b, we retrained the system on **2024 Student Audio Recording** and **2025 Student Audio Recording** datasets respectively.

### Test 1:

In this test, we first played the training samples from the **non-student speech** dataset. Then, we played the testing samples from the same dataset in random order for manual recognition and identity matching. Then, we recorded our manual recognition results as a benchmark accuracy rate to compare with the algorithm results in the following tests to evaluate the algorithm performance.

For the **non-student speech** dataset, the accuracy rate obtained through our human auditory is **62.5%**.

## B. Speech Preprocessing

### Test 2:

In this test, we input each sounds to MATLAB first to check the sampling rate, milliseconds of speech in a block of 256 samples and plot the signal in time domin.

Recordings 9 to 11 are stereo. Therefore, we compute the mean of the two channels to convert them into mono.Also, we applied amplitude normalization to each signal, which is $\frac{y}{max(abs(y))}$.

The sampling rate of each recording reported is 12500 Hz and the milliseconds reported is 20.48 ms.

**Example for the ninth speaker**

This figure shows the original plot for the ninth speaker.

![original](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/signal_9_in_time_domin.png)

This figure shows the plot for the ninth speaker after converting into mono and normalization.

![norm](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/norm_signal_9_in_time_domin.png)

### Use STFT to generate periodogram

In this section, we use Short-time Fourier Transform (STFT) to generate the periodogram for each signal with three different frame sizes: 128, 256, 512. The frame increment M is set as $\frac{N}{3}$ and the window appied is hamming window, which is 

$$w(n)=0.54-0.46cos(2\pi\frac{n}{N}), 0\leq n\leq N$$

#### Example for the first speaker

**Frame size: N=128**

![stft_128](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/stft_128.png)

**Frame size: N=256**

![stft_256](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/stft_256.png)

**Frame size: N=512**

![stft_512](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/stft_512.png)

According to the figures, we found that a lower frame size results in higher time resolution but lower frequency resolution. In contrast, a higher frame size results in lower time resolution but higher frequency resolution.

The results show that the energy is mainly concentrated at low and medium frequencies from 200 ms to 800 ms. Additionally, there is also a significant energy at high frequencies around 220 ms to 330 ms because of the first letter "z" of word "zero."

### Test 3:

In this test, we first generate the mel-spaced filter bank responses using the function *melfb_own* with the parameters: number of filters in filter bank $p=26$, length of fft $n=256$, and sample rate in Hz $fs=12500$. The function *melfb_own* is used to calculate the Mel Filter Bank including Mel scaling, FFT, weighting , with the output filter bank matrix.

![Mel-spaced filter bank responses](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_3_1.png)

The theoretical responses we expected to obtain are triangular. As we can see, the shape of each filter is generally consistent with the theoretical expectation.

Then, we plot the spectrum before and after applying mel-frequency wrapping for the first speaker as the example to explore the impact of this step.

**Example for the first speaker**

![test 3 2](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_3_2.png)

![test 3 3](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_3_3.png)

As we expected, the frequency range is converted to 0–26 Hz, corresponding to 26 Mel-filters. The Mel-frequency wrapping preserved low frequencies components while compressed the high frequencies components to make them smoother. This technology makes the signals more aligned with the characteristics of the human auditory system.

### Test 4:

In this test, we combined all steps above together with the "Cepstrum" to implement the MFCC algorithm. Cepstrum involves the Discrete Cosine Transform (DCT), which is

$$\tilde{c}_n=\sum_1^K (\log \tilde{S}_k)\cos[n(k-\frac{1}{2})\frac{\pi}{K}],    n=0,1,...,K-1,$$

where $$\tilde{c}_n$$ is MFCCs.

After taking the logarithm of the Mel-frequency spectrum, we applied DCT to log Mel-frequency coefficients. The first coefficient was discarded from the results since this coefficient only represents the mean of the signal, which contains little speaker specific information.

### Test 5:

In this test, we used the algorithm previously developed to obtain the MFCC vectors of the first three speakers as an example to check the clustering behavior. The dimensions we chose is the first and second dimensions. In the process of obtaining the MFCC vectors, the first frame consists of the first 100 samples and the second frame begins 100 samples after the first frame, and overlaps it by 156 samples.

**Example for the first three speakers**

![MFCC](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_5.png)

As the figure shows, the points exhibit some clustering but are weak. The points in the central area are densely distributed, and there is significant overlap among the points from the three speakers.

### Test 6:

In order to improve the clustering, we trained a VQ codebook using the LGB algorithm. The algorithm first calculates the centroid of the vectors in the training set to form the initial codebook, and then doubles the size of the codebook by splitting the codebook, which is given by

$$y_n^+=y_n(1+\epsilon),$$
$$y_n^-=y_n(1-\epsilon),$$

where $\epsilon$ is the splitting parameter.

The algorithm employs Nearest Neighbor Search based on Euclidean Distance and iteratively updates the centroids. It compresses the data into several important features based on K-means, which can significantly reduce the confusion caused by unnecessary features and make the data points more structured.

**Example for the first three speakers**

![VQ codewords](https://github.com/dc365710651/EEC-201-FINAL-PROJECT/blob/main/images/test_6.png)

The filled markers are the centroids of each clustering, which represent the features of each speaker.

### Test 7:

In this test we trained our algorithm on the training set from **non-student speech** dataset and then evaluated the accuracy rate on the testing set.

First, we calculate the MFCCs of the training set and generated the codebooks. Then, we computed the MFCCs of the testing set and the distortions between the the MFCCs of the testing set and codebooks of training set. The index of the minimal value of distortions was considered as the prediction of the speaker for the given audio.

The parameters for this test is given below.

| Parameter   | Explanation |   Value  |
|    :---:    |   :----:    |   :---:  |
| K           | Number of vectors in codebook | 17 |
| epsilon     | Splitting parameter | 0.01 |
| threshold   | Convergent condition | 1e-5  |
| max_iter   | Maximum number of iterations | 100 |

Eventually, the accuracy rate of this speaker recognition system on the **non-student speech** dataset is **100%**, which is significantly better than human.

### Test 8:
Accuracy rate = 100%

In the original speaker recognition process, a number of notch filters are added to the test audio, and then the recognition accuracy is measured to verify the stability of the system when some frequency information is lost. In this test, uses MATLAB's iirnotch function to implement notch filtering. Similar to Test7, the VQ codebook is trained by using the training set, and then several test audios are read in turn. The notch filter is performed on each audio, and then the MFCC feature is extracted, compared with the codebook of each speaker, and the recognition accuracy is calculated. 

In this test, we select notchFrequencies = 60, codebookSize = 128, numMelFilters = 26, numCepstra = 12, as a demonstration and got the accuracy rate = 100%.

### Test 9:
Accuracy rate = 94.44%

In this test, we randomly selected speech "zero" of 10 students from 2024 we recorded twice: one for training and one for test. Here we have 18 speakers in total, and codebookSize = 128, numMelFilters = 26, numCepstra = 12, as a demonstration and got the accuracy rate = 94.44%. Compared with the accuracy obtained previously, the output shows that: Test file s15.wav => recognized as Speaker 16, which is a mismatch in additional 'zero'. 


### Test 10_a(Twelve/Zero):

Overall speaker identification accuracy: 91.67%

Overall word identification accuracy: 97.22%

Accuracy for "Twelve" across 18 speakers of 2024 = 94.44%

Accuracy for "Zero" across 18 speakers of 2024 = 88.89%

If we use "twelve" to identify speakers (codebookSize = 128, numMelFilters = 26, numCepstra = 12), the accuracy versus the system that uses "zero" (codebookSize = 32, numMelFilters = 40, numCepstra = 25) improved to 94.44%. There are two mismatches of "Zero": Zero-Word: s6.wav => recognized as Speaker 10, Zero-Word: s11.wav => recognized as Speaker 12. For "Twelve", there is only one mismatch: Twelve-Word: s11.wav => recognized as Speaker 12.

If we train a whole system that tries to identify which speaker and whether the speech is "zero" or "twelve", overall speaker identification accuracy is 91.67%, overall word identification accuracy is 97.22%. In this test, we will test both "Zero" and "Twelve" files. For each test file, we check distances against all codebooks (both Zero and Twelve). Whichever codebook is closest, we take that as the predicted speaker and word. For this test, codebookSize = 32, numMelFilters = 40, numCepstra = 25. From the result, we can find three mismatches: Test file: Zero_test6.wav => Predicted: Speaker 10, Word "Zero", Test file: Twelve_test14.wav => Predicted: Speaker 12, Word "Twelve", Test file: Twelve_test15.wav => Predicted: Speaker 5, Word "Zero".


### Test 10_b(Five/Eleven):

Overall speaker identification accuracy: 97.83%

Overall word identification accuracy: 100.00%

Accuracy for "Five" across 23 speakers of 2025 = 95.65%

Accuracy for "Eleven" across 23 speakers of 2025 = 100.00%

If we use "Eleven" to identify speakers (codebookSize = 128, numMelFilters = 26, numCepstra = 12), the accuracy is 100%. For the test of "Five" (codebookSize = 32, numMelFilters = 40, numCepstra = 25) the accuracy is 95.65%. There is only one mismatch of "Five": Five-Word: s18.wav => recognized as Speaker 13.

If we train a whole system that tries to identify which speaker and whether the speech is "Five" or "Eleven", overall speaker identification accuracy is 97.83%, overall word identification accuracy is 100%. In our system, codebookSize = 128, numMelFilters = 26, numCepstra = 12. There is only one mismatch: Test file: Five s18.wav => Predicted: Speaker 13, Word "Five". Compared with the test in 10a, the accuracy improved a lot.



