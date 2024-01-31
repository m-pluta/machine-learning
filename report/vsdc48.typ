#import "template.typ": *

#show: ieee.with(
  title: [COMP2261 - Machine Learning Report],
  abstract: [
    This study compares k-Nearest Neighbors (kNN) and Logistic Regression for classifying Chinese characters by evaluating the models' accuracy and efficiency in handling high-dimensional data, thereby providing insights into complex character recognition and other digital applications.
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

== Project Summary

The study conclusively demonstrated that Logistic Regression significantly outperforms k-Nearest Neighbors (kNN) in terms of accuracy and inference time when classifying Chinese characters. However, Logistic Regression's longer training time becomes a substantial limitation when dealing with large datasets. The key lesson from this project is the importance of selecting an appropriate algorithm that is not only effective but also scalable, taking into account the size & complexity of the dataset.

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

#v(2pt)

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

The CNN was tuned using a Bandit-Based tuner, Hyperband @Hyperband, provided by the Keras library. Once optimised, the model was cross-validated to ensure major overfitting was not present.


#figure(
  image("img/train_val_cnn.png", width: 100%),
  caption: [
    Graph of training and validation accuracy throughout training the CNN
  ],
) <train_val_accuracy_cnn>

From @train_val_accuracy_cnn it is clear that some overfitting is still present, however, this amount is tolerable. The reason for the model performing better on validation data in the first few epochs is due to the Dropout layer. A dropout layer is a regularisation mechanism that is turned on when training and is turned off when testing. This results in the model performing worse on the training data. @Soleymani_2022 @Keras

= Evaluation Metrics

- *Weighted Accuracy* - Since my dataset has a small amount of imbalance, to be on the safe side I will use the weighted accuracy across all predicted classes instead of average accuracy.
- *F1 Score* - In my classification problem there is no greater negative impact caused by a low sensitivity (Recall). This is not the case with something like medical image classification where low sensitivity could have serious consequences. Therefore, I will use the F1 score which is the harmonic mean of Recall and Precision as they correlate inversely with each other.
- *Training time* - While the total number of Chinese characters remains fairly constant, every so often new characters for complex ideas or newly discovered chemical elements are proposed, and this would warrant retraining/adjusting the model. Lower training times would allow the model to be updated faster.
- *Inference Time* - In applications such as real-time translation, language learning, or even autonomous driving where signs need to be read within fractions of a second, it is crucial to identify text with minimal latency. Additionally, for services with large volumes of data, an efficient, scalable, and high-throughput system is ideal.

= Model Evaluation

== Overview & Justification

For our baseline model, we've selected kNN (k-Nearest Neighbors), a standard, naive algorithm used in classification tasks. Our proposed model is Logistic Regression, chosen for its specific benefits suited to this dataset.

Starting with kNN, it is a non-parametric, lazy-learner. Unlike typical models that learn patterns from training data, kNN instead defers all computations until it's time to make predictions on new data. This characteristic may be advantageous on smaller datasets, but given the large size of our dataset, it could significantly impact the speed of making predictions.

Additionally, kNN suffers a key drawback, the Curse of Dimensionality. The kNN algorithm assumes that similar points in space share the same label @Curse. While this may be the case for smaller dimensions, the feature vectors we obtained in the data exploration section have dimensionality 512, this means that any two vectors belonging to the same class may not be close to each other at all. 

On the other hand, logistic Regression is a parametric model that learns the weights of the features during the training process, which makes it more suitable for handling high-dimensional data @KNN_Logistic_Comparison.

Another key benefit of logistic regression is its efficiency. Unlike kNN which requires storing the entire dataset and searching for the k nearest neighbors on each prediction, logistic regression only needs to apply the learned weights to the test data, likely making predictions much faster especially for large datasets.

== Parameter exploration - kNN

Although kNN is classified as a lazy learner and lacks traditional parameters, it utilizes several hyperparameters including `n_neighbors`, `weights`, `metric`, `p`, `leaf_size`, and `algorithm`. Through experimenting, I found that the hyperparameters `n_neighbors`, `weights`, and `metric` significantly impacted validation accuracy, leading me to focus on their optimization.

I began by first exploring the relationship between `n_neighbors` and validation accuracy.

#figure(
  image("img/knn_k_metric.png", width: 110%),
  caption: [
    Accuracy & F1 Score of K-Nearest Neighbours with varying k
  ],
)

I found that the optimal values for `n_neighbors` lie somewhere between 5 and 20, which allowed me to narrow down the field of possible candidate values. Additionally, I noticed that the f1-score was roughly correlated with the validation accuracy, which led me to believe that the model was effective at not only being accurate but also at maintaining a good balance in avoiding misclassifications.

#figure(
  table(
    columns: (auto, auto, auto),
    align: horizon,
    inset: (
      x: 4pt,
      y: 5pt
    ),
    [], [*Default*], [*Candidate Values*],
    [`n_neighbours`], [`5`], [`range(5, 20)`],
    [`weights`], [`'uniform'`], [`'uniform', 'distance'`],
    [`metric`], [`'minkowski'`], [`'euclidean', 'manhattan', 'minkowski'`],
  ),
  caption: [
    Summary of kNN hyperparameters and candidate values
  ],
)

By performing an exhaustive Grid Search of all the possible configurations, I found that `n_neighbors`=`10`, `weights`=`'distance'`, and `metric`=`'euclidean'` were ideal. 

== Parameter exploration - Logistic Regression

On the other hand, logistic regression does have parameters which are intrinsic to its model. More specifically, logistic regression assigns weights to each feature and adjusts these weights during the training process.

Similarly to kNN, logistic regression also has its hyperparameters, namely `solver`, `C`, `penalty`, `dual`, `tol`, `l1_ratio`. Once again, from experimentation, I found that `solver`, `C`, `penalty`, and `l1_ratio` (if `penalty` was set to `elasticnet`) were the most influential in terms of training time and validation accuracy.

Since my task was concerned with multi-class classification, only the `lbfgs`, `newton-cg`, `sag`, and `saga` solvers supported multinomial loss and so this reduced my field of candidate values.


#figure(
  table(
    columns: (auto, auto, 140pt),
    align: horizon,
    inset: (
      x: 4pt,
      y: 5pt
    ),
    [], [*Default*], [*GridSearch*],
    [`solver`], [`'lbfgs'`], [`'lbfgs', 'newton-cg', 'sag', 'saga'`],
    [`penalty`], [`'l2'`], [`'l1', 'l2', 'elasticnet', 'None'`],
    [`C`], [`1`], [`10, 5, 1, 0.5, 0.1`],
    [`l1_ratio` \ (saga only)], [`None`], [`0.25, 0.5, 0.75`],
  ),
  caption: [
    Summary of hyperparameters for \ Logistic Regression and candidate values
  ],
)

Similarly, by performing an exhaustive Grid Search of all the possible configurations, I found that `solver`=`lbfgs`, `C`=`5`, and `penalty`=`l2` were ideal. 

#figure(
  image("img/model_eval.svg", width: 90%),
  caption: [
    Accuracy vs Training Accuracy of Logistic Regression with varying amounts of training samples
  ],
)

Upon comparing the performance of both models against each other with varying amounts of training data, it was clear that logistic regression was much more accurate at classifying and was on average 8% more accurate (92% vs 84%). Unsurprisingly, it was also clear that the kNN model was much faster at training. This was essentially because kNN's 'training' is just loading the feature vectors into memory. Most importantly, the logistic regression model had a much lower inference time, which is solely attributed to the way it learns the training data and updates its weights. In theory, this makes the logistic regression model much more scalable.

// #figure(
//   image("img/wrong_predictions.png", width: 90%),
//   caption: [
//     Distribution of image dimensions in the original dataset
//   ],
// )

#v(-5pt)

= Self Evaluation

== Lectures

Throughout the lectures, I was most intrigued by the mathematical concepts that underpin each machine-learning model. It made me glad to know that the mathematical foundations I had practiced tirelessly last year were crucial to understanding both the inner workings of each model as well as the intuition behind how they all were developed. The lectures also showed me that not all models behave like a 'black box', and many have visual representations such as kNN and Lloyd's Algorithm in KMeans.

== Coursework

One particular thing this coursework has taught me is the machine learning workflow. The lectures didn't go into much detail about the practical aspect, however, this coursework was a very good introduction on how to approach a machine learning problem. From data preprocessing and feature selection to model training and evaluation, I learned to consider each step critically. Most importantly this coursework highlighted the iterative nature of machine learning, where models are continuously refined based on performance metrics.

== Module difficulties

My main difficulty in the module is being able to effectively translate my theoretical understanding into a practical solution. Ultimately, most of this content is completely new to me, hence I've never had a chance to implement or tune any models independently. Thankfully, the practicals were able to address some of these concerns as I got hands-on experience implementing and adjusting various models.

== Reflection

I believe the biggest challenge of this assignment was the size of the dataset and the disproportionate computational power I possessed. I approached this project wanting to do image classification related to one of my interests, however, I didn't fully consider how long it would take to train even a single model with such a vast quantity of data. Moreover, I believe it would have been more effective to use a pre-trained CNN such as Resnet to ensure the feature extraction was as accurate as possible since all the other models rely on this.

== Unique contributions

While my models do not constitute anything novel, I find it nevertheless an area of research that should and likely will be explored.

#v(-15pt)