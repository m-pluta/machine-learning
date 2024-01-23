#import "template.typ": *

#show: ieee.with(
  title: [COMP2261 - Machine Learning Report],
  abstract: [
    #lorem(20)
  ],
  authors: (
    (
      name: "Michal Pluta",
    ),
  ),
  index-terms: ("Machine Learning", "K-nearest-Neighbours", "Logistic Regression", "Hyperparameter Tuning, CNN"),
  bibliography-file: "refs.bib",
)

#set list(
  marker: [•],
  indent: 6pt,
  body-indent: 6pt
)

= Introduction

 Mandarin Chinese is one of the most spoken languages with 1.3 billion native speakers globally. In countries with significant Chinese-speaking populations, the ability to accurately and automatically identify characters has applications across a variety of domains. To name a few:

- Recognising mail addresses in the postal service, reducing the dependency on manual sorting.
- Digitisation of cultural heritage, where manual digitisation is too difficult or infeasible time-wise.
- Improving accessibility for the visually impaired or those who have reading difficulties.

This report explores the application of machine learning methods on a dataset comprising Chinese (Simplified) characters, akin to the widely recognised MNIST dataset for handwritten digits. The significance of this study lies in the inherent complexity and variety of Chinese script, and the motivation lies in a personal interest in learning the language. 

#figure(
  image("img/ml_system.png", width: 90%),
  caption: [
    Workflow of the proposed ML System
  ],
)

The primary objective is to demonstrate the performance, tuning process, and limitations of two separate machine learning models.

*#lorem(110)*

= Dataset Overview

While Chinese has more than 50,000 characters with roughly 6,500 in daily use, we will only deal with a subset part of the HSK 1 curriculum. HSK is a Chinese Proficiency Exam with levels ranging from 1 (Beginner) to 9 (Near-native).

Chinese characters, also known as Hanzi, are composed of strokes, the basic units of writing, arranged in a specific order and direction. The way these strokes combine give rise to a vast array of characters, each with its own unique meaning and structure.

#figure(
  image("img/example_images.png", width: 90%),
  caption: [
    Examples of images in the dataset
  ],
) <ExampleImages>

The uniqueness of handwritten Chinese characters becomes even more pronounced when considering China's population. This alone leads to a huge variation in style, consistency, size, alignment, and thickness of strokes (@ExampleImages).

= Dataset transformations

The original dataset @Dataset contains 178 classes of images, split using an 80/20 train-test split. The images are greyscale with a 1:1 aspect ratio, and in total there are 131,946 samples. To minimise bias and to ensure the train and test sets were representative of the whole dataset, the sets were merged and then shuffle-split through stratification. This also allowed the dataset to be standardised using one directory only.

== Image Resizing

One clear issue with the original dataset were the varying image dimensions.

#figure(
  image("img/image_dimension_distribution.png", width: 90%),
  caption: [
    Distribution of image dimensions in the original dataset
  ],
)

This was a problem because the CNN used for feature extraction has specific input tensor dimension requirements, hence, all images had to be standardised. $48 times 48$ was chosen as a suitable image size as it meant most images could be downsampled. Downsampling is often a better technique than upsampling as it doesn't limit the model's ability to learn key features like object boundaries. Additionally, due to the complexity of Chinese characters, the images could not be downsampled further due to the likely loss of information.

#figure(
  image("img/class_imbalance.svg", width: 90%),
  caption: [
    Distribution of the number of samples in each image class.
  ],
)

From further exploration, a class imbalance was evident.

#figure(
  table(
    columns: (auto, auto),
    inset: (
      x: 10pt,
      y: 6pt
    ),
    [*Imbalance metric*], [*Value*],
    [Imbalance Ratio], [1.03745],
    [Interquartile Range], [3.0],
    [Coefficient of Variation], [0.00409]
  ),
  caption: [
    Dataset imbalance metric values
  ],
)

Evaluating the imbalance ratio, IQR, and coefficient of variation, it was clear the imbalance was minimal and insignificant.

== Formal Definition

More formally, the images $x^"(i)"$, and one-hot encoded labels $y^"(i)"$ in this system are defined as follows:


$ x^"(i)" in RR^(48 times 48) $
$ y^"(i)" in RR^(178) $

By implication, the input tensors are defined as \

#set align(center)

$ x in RR^(131946 times 48 times 48) $

$ y in RR^(131946 times 178) $

#set align(left)

This allows us to define the CNN's feature extraction operation, $"extract"()$ as

$ "extract"(x^"(i)") = f^"(i)" in RR^512 $

== Feature Extraction

Since the dataset was comprised of images, relevant features had to be extracted from each image. One of the most effective ways of doing this is by using a CNN (Convolutional Neural Network).

While pre-trained CNNs such as Resnet50 were initially considered due to their simplicity and outstanding performance, they yielded feature vectors which were simply too large (2048), resulting in unmanageable training times. One way to handle this would be to perform dimensionality reduction using a technique such as PCA, however, this is just as, if not more computationally intensive.

Instead, a custom CNN was developed using Keras' deep-learning library. The model features 3 2D-Convolution layers, each followed by a MaxPooling layer. While developing the model, severe overfitting was noticed. This was addressed by adding Dropout layers to the model, which would randomly set a fraction of the input units to 0. This improved the model's ability to generalise.

The CNN was tuned using a Bandit-Based tuner, Hyperband, provided by the Keras library. Once optimised, the model was cross-validated to ensure major overfitting was not present.

#figure(
  image("img/train_val_cnn.png", width: 120%),
  caption: [
    Graph of training and validation accuracy throughout training the CNN
  ],
) <train_val_accuracy_cnn>

From @train_val_accuracy_cnn it is clear that some overfitting is still present, however, this amount is tolerable. The reason for the model performing better on validation data in the first few epochs is due to the Dropout layer. While the model is 





// The model was trained on validation data and achieved an average validation accuracy of 94.2%.





= Evaluation Metrics

- *Weighted Accuracy* - Since my dataset has a small amount of imbalance, to be on the safe side I will use the weighted accuracy across all predicted classes instead of average accuracy.
- *F1 Score* - In my classification problem there is no greater negative impact caused by a low sensitivity (Recall). This is not the case with something like medical image classification where low sensitivity could have serious consequences. Therefore, I will use the F1 score which is the harmonic mean of Recall and Precision as they correlate inversely with each other.
- *Training time* - While the total number of Chinese characters remains fairly constant, every so often new characters for complex ideas or newly discovered chemical elements are proposed, and this would warrant retraining/adjusting the model. Lower training times would allow the model to be updated faster.
- *Inference Time* - In applications such as real-time translation, language learning, or even autonomous driving where signs need to be read within fractions of a second, it is crucial to identify text with minimal latency. Additionally, for services with large volumes of data, an efficient, scalable, and high-throughput system is ideal.

= Model Evaluation

*#lorem(100)*

#figure(
  image("img/model_eval.svg", width: 90%),
  caption: [
    Accuracy of K-Nearest Neighbours for varying k
  ],
)

*#lorem(100)*

#figure(
  table(
    columns: (92pt, 65pt, 67pt),
    align: horizon,
    inset: (
      x: 8pt,
      y: 5pt
    ),
    [*Hyperparameter*], [*Default*], [*GridSearch *],
    [`n_neighbours`], [`5`], [`10`],
    [`weights`], [`'uniform'`], [`'distance'`],
    [`metric`], [`'minkowski'`], [`'euclidean'`],
  ),
  caption: [
    Summary of tuned hyperparameters for \ k-Nearest-Neighbours
  ],
)

#figure(
  table(
    columns: (92pt, 65pt, 67pt),
    align: horizon,
    inset: (
      x: 8pt,
      y: 5pt
    ),
    [*Hyperparameter*], [*Default*], [*GridSearch*],
    [`solver`], [`'lbfgs'`], [`'lbfgs'`],
    [`penalty`], [`'l2'`], [`'l2'`],
    [`C`], [`1`], [`5`],
  ),
  caption: [
    Summary of tuned hyperparameters for \ Logistic Regression
  ],
)


*#lorem(100)*

#figure(
  image("img/wrong_predictions.png", width: 90%),
  caption: [
    Distribution of image dimensions in the original dataset
  ],
)

#figure(
  image("img/knn_k_metric.png", width: 90%),
  caption: [
    Accuracy vs Training Accuracy of Logistic Regression with varying amounts of training samples
  ],
)



= Self Evaluation

== Lectures

Throughout the lectures, I was most intrigued by the mathematical concepts that underpin each machine-learning model. It made me glad to know that the mathematical foundations I had practiced tirelessly last year were crucial to understanding both the inner workings of each model as well as the intuition behind how they all were developed. The lectures also showed me that not all models behave like a 'black box', and many have visual representations such as kNN and Lloyd's Algorithm in KMeans.

== Coursework

One particular thing this coursework has taught me 

== Module difficulties

My main difficulty in the module is being able to effectively translate my theoretical understanding into a practical solution. Ultimately, most of this content is completely new to me, hence I've never had a chance to implement or tune any models independently. Thankfully, the practicals were able to address some of these concerns as I got hands-on experience implementing and adjusting various models.

== Reflection

I believe the biggest challenge of this assignment was the size of the dataset and the disproportionate computational power I possessed. I approached this project wanting to do image classification related to one of my interests, however, I didn't fully consider how long it would take to train even a single model with such a vast quantity of data. Moreover, I believe it would have been more effective to use a pre-trained CNN such as Resnet to ensure the feature extraction was as accurate as possible since all the other models rely on this.

== Unique contributions

While my models do not constitute anything novel, I find it nevertheless an area of research that should be explored. 