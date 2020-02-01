CIFAR-10 Image Classification

<!-- TOC -->

- [1. The Dataset](#1-the-dataset)
- [2. Feature Extraction](#2-feature-extraction)
  - [2.1. Histogram of Oriented Gradients](#21-histogram-of-oriented-gradients)
  - [2.2. Local Binary Patterns](#22-local-binary-patterns)
  - [2.3. Harris Corner](#23-harris-corner)
- [3. Model Construction](#3-model-construction)
  - [3.1. Logistic Regression](#31-logistic-regression)
  - [3.2. XGBoost](#32-xgboost)
  - [3.3. Multi-layer Perceptron](#33-multi-layer-perceptron)
  - [3.4. An Ordinary Deep CNN](#34-an-ordinary-deep-cnn)
  - [3.5. ResNet20](#35-resnet20)
- [4. Comparison](#4-comparison)
- [5. Further Exploration](#5-further-exploration)
- [Dependences](#dependences)

<!-- /TOC -->

# 1. The Dataset

Cifar-10 is a data set collected by Alex Krizhevsky, Vinod Nair, Geoffrey Hinton. The dataset consists of 60000 3×32×32 color images in 10 classes, with 6000 images per class. The images are divides into:

1. 50000 training images

    Trainset contains exactly 5000 images from each class, randomly divided into five training batches. Some training batches may contain more images from one class than another.

1. 10000 test images

    Testset contains exactly 1000 randomly-selected images from each class.

The 10 classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. Providers of the dataset offer an official benchmark for model trainers: 18% error rate. Our goal for this task is to exceed the benchmark.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/cifar10.png)

# 2. Feature Extraction

There are many different kinds of image features that we use for our classification task. In this task, we mainly extract three kinds of features: Histogram of Oriented Gradients, Local Binary Patterns and Harris Corner.

## 2.1. Histogram of Oriented Gradients

The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. In the HOG feature descriptor, the distribution (histograms) of directions of gradients (oriented gradients) are used as features. Gradients (x and y derivatives) of an image are useful because the magnitude of gradients is large around edges and corners (regions of abrupt intensity changes) and we know that edges and corners pack in a lot more information about object shape than flat regions.

The HOG descriptor maintains a few key advantages over other descriptor methods. Since the HOG descriptor operates on localized cells, the method upholds invariance to geometric and photometric transformations, except for object orientation. Such changes would only appear in larger spatial regions. However, the disadvantage is that the final descriptor vector grows larger, thus taking more time to extract and to train using a given classifier.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/hog.png)

## 2.2. Local Binary Patterns

Local binary patterns (LBP) is a type of visual descriptor used for classification in computer vision. For each pixel in a cell, compare the pixel to each of its 8 neighbors (on its left-top, left-middle, left-bottom, right-top, etc.). Follow the pixels along a circle, i.e. clockwise or counter-clockwise. Where the center pixel&#39;s value is greater than the neighbor&#39;s value, write &quot;0&quot;. Otherwise, write &quot;1&quot;.

LBP has high discriminative power, computational simplicity and invariance to grayscale changes. However, LBP is not invariant to rotations. The size of the features increases exponentially with the number of neighbors which leads to an increase of computational complexity in terms of time and space. And the structural information captured by it is limited. Only pixel difference is used, magnitude information ignored.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/lbp.png)

## 2.3. Harris Corner

Harris Corner Detector is a kind of corner detection operator (similar to Kitchen and Rosenfeld corner, Shi-Tomasi corner) that is commonly used in computer vision algorithms to extract corners and infer features of an image. Harris corner uses Harris response calculation.

Harris corner detector provides good repeatability under changing illumination and rotation, and therefore, it is more often used in stereo matching and image database retrieval. It also possesses drawbacks like smoothing out of weak corners and degraded localization accuracy.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/ch.png)

# 3. Model Construction

In our task, we tried different models to deal with the classification problem. The models can be divided into two types: shallow models and deep models. The trick we use here is to extract the features and train the shallow model, but train the deep model without features because the deep convolutional neural network will extract the features itself.

For the shallow models, we use the three features to train the model separately. Then we combine the three features together to train. Lastly, we train the model use the original pixels (without features) as the benchmark evaluation for different features. If the accuracy under a feature is lower than the benchmark, there is no need to extract that feature.

For the deep models, we apply a technique called learning rate scheduler to improve the accuracy after the model is overfitted under a constant learning rate and the technique works well.

## 3.1. Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. It can also deal with multi-class task and penalty.

In our model, we add a &quot;L2&quot; penalty to the model, use the LBFGS algorithm to do the optimization and set the maximum iterations to 1000.

Table 1 provides the results of logistic regression. Here, HOG performs best and takes the lowest time to train. But the model is too simple so that the accuracy even doesn&#39;t exceed 50%.

>Table 1 Results of Logistic Regression

| Feature | Accuracy on testset | Train duration (s) |
| --- | --- | --- |
| Original pixels (benchmark) | 0.3735 | 162 |
| HOG | 0.4864 | 8.1 |
| Corner Harris | 0.3153 | 72 |
| Local Binary Pattern | 0.2303 | 78 |
| Combined features | 0.3676 | 132 |

## 3.2. XGBoost

XGBoost is a decision-tree-based ensemble machine learning algorithm that uses a gradient boosting framework. XGBoost has been utilized in many data science competitions and performs well on structured data. Therefore, we can foresee that XGBoost may not be suitable for this image task.

Table 2 gives out the results. As we can see, the results prove our prediction. XGBoost took a lot of time to train the model but only improve the accuracy by around 7%. Thus, we shouldn&#39;t use a tree-based model on an unstructured data task.

>Table 2 Results of XGBoost

| Feature | Accuracy on testset | Train duration |
| --- | --- | --- |
| Original pixels (benchmark) | 0.4829 | 31.8m |
| HOG | 0.5406 | 4.7m |
| Corner Harris | 0.4783 | 14.8m |
| Local Binary Pattern | 0.3156 | 7.4m |
| Combined features | 0.5554 | 25.7m |

## 3.3. Multi-layer Perceptron

A multilayer perceptron is a subset of artificial neural network. We establish a multilayer perceptron only with one hidden layer.

As we can see in the previous results, LBP really reaches a bad score. So, we combine only HOG and Harris corner together and train an additional model.

Specifically, we use 150 nodes in the hidden layer, use the ReLU activation function. We set the learning rate to 0.001 and the maximum iteration to 200.

Table 3 shows the results. HOG is still the best feature we have and the accuracy first exceeds 60%, which is an acceptable threshold for practical implementation of the model. And as we expected, combined features without LBP performs better than all the features combined.

>Table 3 Results of Multi-layer Perceptron

| Feature | Accuracy on testset |
| --- | --- |
| Original pixels (benchmark) | 0.4649 |
| HOG | 0.6164 |
| Corner Harris | 0.3416 |
| Local Binary Pattern | 0.2207 |
| Combined features | 0.4559 |
| Combine only HOG&amp;CH | 0.4814 |

## 3.4. An Ordinary Deep CNN

Then we come to the deep models.

We establish an ordinary convolutional neural network following the ordinary order of layers like convolutional layer - max pooling layers - dropout layers. The architecture of our CNN model is shown by figure 6, generated by the python package Keras that we use to write the codes.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/cnn.png)

In this model, we have 1,250,858 trainable parameters. We set the learning rate as 0.1 and the maximum iteration as 100 to train the first CNN model. Table 4 offers the results. We can see that the accuracy on testset has been hugely increased from around 60% to 80%.

>Table 4 Result of the first CNN

| Accuracy on trainset | Accuracy on testset | Train duration |
| --- | --- | --- |
|  0.9082 | 0.8027 | 34.9 min |

In order to improve the model, we look deeply into the training process. Figure 7 shows how the loss and accuracy evolve during the 100 iterations. As we can see in the line chart, the loss of trainset keeps decreasing and the accuracy of trainset keeps increasing during all over the training process. However, the loss and accuracy for testset seems to stop changing after tens of iterations. This phenomenon indicate that the model faces overfitting problem, which means the improvement on trainset will not help the testset anymore.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/cnnloss.png)

Under this circumstance, we figure out a trick to help the model to learn new information after the edge of overfitting. The trick is to schedule the learning rate during the training process, to decrease the learning rate after certain number of iterations. We set the learning rate as table 5.

>Table 5 Scheduler of Learning Rate

| Iteration | <80 | <120 | <160 | <180 | <200 |
| --- | --- | --- | --- | --- | --- |
| Learning rate | 0.001 | 0.0001 | 1e-5 | 1e-6 | 5e-7 |

The result and evolvement of accuracy are provided in table 6 and figure 8. As we can see, there is an obvious increase to accuracy after the learning rate has been changed from 0.001 to 0.0001. And it really helps to improve a good model to be even better.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/cnnacc.png)

>Table 6 Result of the scheduled CNN

| Accuracy on trainset | Accuracy on testset | Train duration |
| --- | --- | --- |
| 0.9585 | 0.8186 | 35.8 min |

## 3.5. ResNet20

Finally, we come to the finely designed deep network ResNet, which was proposed by researches of Microsoft in 2015. This network won first of the ImageNet classification competition in 2015.

The ResNet series have different networks with different depth. They all base on the residual network blocks.

The residual block helps the model to gain information left from the former layer, thus increase the fluctuation but gain more information.

We train the shallowest one the ResNet series, ResNet20. Is has 274,442 parameters and 273,066 trainable ones. Likewise, we use the same learning rate scheduler trick on this training process.

Results are offered by table 7. As we can see, the test accuracy fluctuates much more than the ordinary CNN due to the usage of residual blocks. And the increase of accuracy after changing the learning rate exists as well. ResNet20 achieves 84.39% of the testset, which has exceeded the benchmark 82%. So far, we have reached our goal for this task.

![](https://github.com/CaptainCandy/classification_cifar10/blob/master/images/resacc.png)

>Table 7 Results of ResNet20

| Accuracy on trainset | Accuracy on testset | Train duration |
| --- | --- | --- |
| 1 | 0.8439 | 116m |

# 4. Comparison

In summary, we compare the models we&#39;ve used in this task on table 8. We used two main method to train the classification model.

The first is traditional features plus shallow classifiers. Under this method, HOG feature performs best among the three independent features. Sometimes the combined features did a good job (XGBoost) but sometimes not. Multi-layer perceptron performs the best among the three shallow models.

The second method is to use deep convolutional neural network. We find that Even an ordinary CNN can gain a relatively high accuracy score. The trick of learning rate scheduler can help if we want to chase the ultimately high accuracy. Meanwhile, Fine designed ResNet earns the best score, but it also is time-consuming.

>Table 8 Model Comparison

| Accuracy | Logistic regression | XGBoost | Multi-layer perceptron | CNN | ResNet20 |
| --- | --- | --- | --- | --- | --- |
| Original pixels | 0.3735 | 0.4829 | 0.4649 | 0.8186 | 0.8439 |
| HOG | 0.4864 | 0.5406 | 0.6164 |   |   |
| Combined | 0.3676 | 0.5554 | 0.4559 |   |   |

# 5. Further Exploration

Since the dataset contains 10 classes of objects, we shouldn&#39;t only focus on the total accuracy of the model. Thus, we do a further exploration on per-class accuracy.

As we can see in table 9, the accuracy of different classes differs. Accuracy of ship is the highest and cat is the lowest.

>Table 9 Per-class Accuracy

| Classes | CNN | ResNet20 |
| --- | --- | --- |
| Airplane | 0.848 | 0.844 |
| Automobile | 0.902 | 0.817 |
| Ship | 0.904 | 0.947 |
| Truck | 0.903 | 0.838 |
| Bird | 0.703 | 0.688 |
| Cat | 0.660 | 0.682 |
| Deer | 0.814 | 0.842 |
| Dog | 0.737 | 0.658 |
| Frog | 0.893 | 0.820 |
| Horse | 0.829 | 0.856 |

When we group the ten classes to two big classes: vehicle and animal, we find that accuracy for vehicle is around 10% higher than animal (see in table 10).

In additional, we can tell that different models behave differently. Ordinary CNN does 8% better classification on dogs than ResNet20.

>Table 10 Two Classes Accuracy

| Classes | CNN | ResNet20 |
| --- | --- | --- |
| Vehicle | 0.889 | 0.862 |
| Animal | 0.773 | 0.758 |

# Dependences
- python 3.5.6
- keras 2.0.8
- opencv 3.4.1
- numpy 1.14.2
- scikit-image 0.14.0
- scikit-learn 0.20.0
- matplotlib 2.2.3
