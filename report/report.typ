#import "template.typ": *
#set list(
  marker: [•],
  indent: 8pt,
  body-indent: 8pt
)

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


= Introduction

In this report, we will explore the application of deep learning & machine learning methods on a dataset comprising of Chinese (Simplified) characters, akin to the widely recognised MNIST dataset for handwritten digits. The significance of this study lies in the inherent complexity and variety in Chinese script, and the motivation lies in a personal interest of learning the language. The primary objective is to demonstrate the performance, tuning process, and limitations of two seperate machine learning models. 

Mandarin Chinese is one of the most spoken languages with 1.3 billion native speakers globally. In countries with significant Chinese-speaking populations, the ability to accurately and automatically identify characters has applications across a variety of domains. To name a few:

- Recognising mail addresses in the postal service, reducing the dependency on manual sorting.
- Digitisation of cultural heritage, where manual digitisation is too difficult or infeasible time-wise.
- Improving accessibility for the visually impaired or those who have reading difficulties.

*#lorem(190)*


= Dataset Overview

While Chinese has more than 50,000 characters with roughly 6,500 in daily use, we will only deal with a subset part of the HSK 1 curriculum. HSK is Chinese Proficiency Exam with levels ranging from HSK 1 (Beginner) up to HSK 9 (Near native). 

Chinese characters, also known as Hanzi, are composed of strokes, the basic units of writing, arranged in a specific order and direction. The way these strokes are combined gives rise to a vast array of characters, each with its own unique meaning and structure.

The uniqueness of handwritten Chinese characters becomes even more pronounced when considering China's population. This alone leads to a huge variation in style, consistency, size, alignment, and thickness of strokes.

#figure(
  image("img/example_images.png", width: 90%),
  caption: [
    Examples of images in the dataset
  ],
)

= Dataset transformations

The original dataset contains 178 classes of images, split using an 80/20 train-test split. To ensure the train and test sets were representative of the whole dataset and unbiased, the sets were merged and then shuffle-split again as a pre-processing step. This also meant the dataset could be standardised by working with one directory only.

== Image Resizing

One evident issue with the original dataset were the varying image dimensions.

#figure(
  image("img/image_dimension_distribution.png", width: 90%),
  caption: [
    Distribution of image dimensions in original datset
  ],
)

This was a problem because the CNN used for feature extraction has specific input tensor dimension requirements, hence, all images had to be standardised. $64 times 64$ was chosen as a suitable image size as it meant most images could be downsized. Downsizing is often a better technique than increasing size as it doesn't limit the model's ability to learn key features like object boundaries. While resizing, anti-aliasing was turned on.

== Formal Definition

*#lorem(100)*

== Feature Extraction

*#lorem(100)*

#figure(
  image("img/train_val_accuracy_cnn.png", width: 90%),
  caption: [
    Distribution of image dimensions in original datset
  ],
)

*#lorem(200)*

= Evaluation Metrics

*#lorem(140)*

= Model Evaluation

*#lorem(200)*

*#lorem(100)*

*#lorem(150)*

#figure(
  image("img/wrong_predictions.png", width: 90%),
  caption: [
    Distribution of image dimensions in original datset
  ],
)

*#lorem(150)*


= Self Evaluation

== Lectures

Throughout the lectures I have been most intrigued by the mathematical concepts that underpin each machine learning model. It made me happy knowing that the mathematical foundations I had practiced tirelessly last year were crucial to understanding both the inner workings of each model as well as the intuition behind how they all are developed.




== Coursework

*#lorem(100)*

== Module difficulties

My main difficulty in the module is being able to effectively translate my theoretical understanding into a practical solution. Ultimately, most if not all of this content is completely new to me so I've never had a chance to implement or tune any models independently. Thankfully, the practicals were able to address some of these concerns as I got hands-on experience implementing and adjusting various models.

== Reflection

Reflecting back on this assignment, I believe one of the biggest challenges was the size of the dataset and the lack of computational power I possessed. I approached this project wanting to do image classification related to one of my interests, but didn't fully consider how long it would take to train even a single model with such a vast quantity of data.

== Unique contributions

*#lorem(40)*