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

This report explores the application of machine learning methods on a dataset comprising Chinese (Simplified) characters, akin to the widely recognised MNIST dataset for handwritten digits. The significance of this study lies in the inherent complexity and variety of Chinese script, and the motivation lies in a personal interest in learning the language. The primary objective is to demonstrate the performance, tuning process, and limitations of two separate machine learning models.

#figure(
  image("img/ml_system.png", width: 90%),
  caption: [
    Workflow of the proposed ML System
  ],
)

\

*#lorem(160)*

= Dataset Overview

While Chinese has more than 50,000 characters with roughly 6,500 in daily use, we will only deal with a subset part of the HSK 1 curriculum. HSK is a Chinese Proficiency Exam with levels ranging from HSK 1 (Beginner) up to HSK 9 (Near-native).

Chinese characters, also known as Hanzi, are composed of strokes, the basic units of writing, arranged in a specific order and direction. The way these strokes are combined gives rise to a vast array of characters, each with its unique meaning and structure. The uniqueness of handwritten Chinese characters becomes even more pronounced when considering China's population. This alone leads to a huge variation in style, consistency, size, alignment, and thickness of strokes.

#figure(
  image("img/example_images.png", width: 90%),
  caption: [
    Examples of images in the dataset
  ],
)

= Dataset transformations

The original dataset @Dataset contains 178 classes of images, split using an 80/20 train-test split. The images were all greyscale with a 1:1 aspect ratio, and in total, there were 131,946 images. To minimise bias and to ensure the train and test sets were representative of the whole dataset, the sets were merged and then shuffle-split through stratification. This also meant the dataset could be standardised by working with one directory only.

== Image Resizing

One evident issue with the original dataset were the varying image dimensions.

// #figure(
//   image("img/image_dimension_distribution.svg", width: 100%),
//   caption: [
//     Distribution of image dimensions in the original dataset
//   ],
// )

This was a problem because the CNN used for feature extraction has specific input tensor dimension requirements, hence, all images had to be standardised. $48 times 48$ was chosen as a suitable image size as it meant most images could be downsampled. Downsampling is often a better technique than upsampling as it doesn't limit the model's ability to learn key features like object boundaries. Additionally, due to the complexity of Chinese characters, the image dimensions could not be reduced further due to loss of information.

// #figure(
//   image("img/class_imbalance.svg", width: 100%),
//   caption: [
//     Distribution of the number of samples in each image class.
//   ],
// )

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


== Feature Extraction

*#lorem(100)*

#figure(
  image("img/train_val_accuracy_cnn.png", width: 90%),
  caption: [
    Distribution of image dimensions in the original dataset
  ],
)

*#lorem(200)*

== Formal Definition

More formally, the images $x^"(i)"$, and one-hot encoded labels $y^"(i)"$are defined as follows:


$ x^"(i)" in RR^(48 times 48) $
$ y^"(i)" in RR^(178) $

By implication, the input tensors are defined as \

#set align(center)

$ x in RR^(131946 times 48 times 48) $

$ y in RR^(131946 times 178) $

#set align(left)

This allows us to define the CNN's feature extraction operation, $"extract"()$ as

$ "extract"(x^"(i)") = f^"(i)" in RR^512 $


= Evaluation Metrics

- *Weighted Accuracy* - Since my dataset has a small amount of imbalance, to be on the safe side I will use the weighted accuracy across all predicted classes instead of average accuracy.
- *F1 Score* - In my classification problem there is no greater negative impact caused by a low sensitivity (Recall). This is not the case with something like medical image classification where low sensitivity could have serious consequences. Therefore, I will use the F1 score which is the harmonic mean of Recall and Precision as they correlate inversely with each other.
- *Training time* - While the total number of Chinese characters remains fairly constant, every so often new characters for complex ideas or newly discovered chemical elements are proposed, and this would warrant retraining/adjusting the model. A faster training would allow the model to updated quicker.
- *Inference Time* - In applications such as real-time translation, language learning, or even autonomous driving where signs need to be read within fractions of a second, it is crucial to identify text with minimal latency. Additionally, for services with large volumes of data, an efficient, scalable, and high-throughput system is ideal.

= Model Evaluation

*#lorem(200)*

// #figure(
//   image("img/filler.svg", width: 100%),
//   caption: [
//     Accuracy of K-Nearest Neighbours for varying k
//   ],
// )

*#lorem(100)*

#figure(
  table(
    columns: (auto, auto, auto),
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
    Summary of tuned hyperparameters for \ k-Nearest-Neighours
  ],
)

#figure(
  table(
    columns: (auto, auto, auto),
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


*#lorem(230)*

// #figure(
//   image("img/wrong_predictions.png", width: 90%),
//   caption: [
//     Distribution of image dimensions in the original dataset
//   ],
// )

// #figure(
//   image("img/filler.svg", width: 100%),
//   caption: [
//     Accuracy vs Training Accuracy of Logistic Regression with varying amounts of training samples
//   ],
// )

*#lorem(250)*


= Self Evaluation

== Lectures

Throughout the lectures, I was most intrigued by the mathematical concepts that underpin each machine-learning model. It made me glad to know that the mathematical foundations I had practiced tirelessly last year were crucial to understanding both the inner workings of each model as well as the intuition behind how they all were developed. The lectures also showed me that not all models behave like a 'black box', and many have visual representations such as kNN and Lloyd's Algorithm in KMeans.

== Coursework

*#lorem(70)*

== Module difficulties

My main difficulty in the module is being able to effectively translate my theoretical understanding into a practical solution. Ultimately, most of this content is completely new to me, hence I've never had a chance to implement or tune any models independently. Thankfully, the practicals were able to address some of these concerns as I got hands-on experience implementing and adjusting various models.

== Reflection

I believe the biggest challenge of this assignment was the size of the dataset and the disproportionate computational power I possessed. I approached this project wanting to do image classification related to one of my interests, however, I didn't fully consider how long it would take to train even a single model with such a vast quantity of data. Moreover, I believe it would have been more effective to use a pre-trained CNN such as Resnet to ensure the feature extraction was as accurate as possible since all the other models rely on this.

== Unique contributions

While my models do not constitute anything novel, I find it nevertheless an area of research that should be explored. 