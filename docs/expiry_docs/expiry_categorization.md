Expiry Categorization
---------------------

## Overview

In the field of computer vision, *segmentation* refers to isolating relevant elements from an overall image, while *categorization* refers to examining a particular element and determining what it represents.

As described in [Expiry Segmentation](expiry_segmentation.md), card.io's segmentation step for expiry isolates individual characters that are likely to belong to an expiration date. Probable slash characters (`/`) are categorized as part of the segmentation step. Each of the remaining characters is then passed through the main expiry categorization step to classify it as a particular digit.

### Deep Learning models

Categorizing expiry digits is challenging! The characters are small, measuring only about 9x15 pixels at standard camera resolution. (See [Camera resolution](../camera_resolution.md).) Card backgrounds vary widely, as do the effects of various lighting conditions.

We use deep-learning neural-net models to perform categorization. The input to a model is a likely character image. The output is a set of category probabilities which sum to 1. For the slash categorizer, there are two categories: slash or non-slash. For the digit categorizer there are 10 categories: one for each digit.

We create our models using [Theano](http://www.deeplearning.net/software/theano/), combined with a library of Python utilities that we have created for our own use. For expiry categorization, we use [multilayer perceptron](http://www.deeplearning.net/tutorial/mlp.html#mlp) and [convolutional](http://www.deeplearning.net/tutorial/lenet.html#lenet) models. We began with the sample Theano code, but have since added a number of refinements.

We have built and trained *many* models with a great variety of characteristics (depth, breadth, etc.). The choices of which models to include in card.io are based on each model's accuracy, constrained by the speed and size of the models. Trying to maintain a reasonable total size for this mobile-device library significantly limits the complexity of models available to us.

Models are trained with images captured from a set of real credit cards which the card.io team has collected over several years. Unfortunately, this adds another limitation, as the range of expiration dates is narrow. For example, we have many, many images of the digits `0` and `1`, but rather few `5`s. Given the nature of training neural nets, our models will therefore produce more accurate results when presented with a `0` or `1` than when shown a `5`.

Some of the deep learning training techniques we've used, including some not described in the [Theano tutorials](http://www.deeplearning.net/tutorial/) are:

* early stopping [[1]](#1)
* dropout [[2](#2), [3](#3), [4](#4)]
* normalizing inputs by subtracting the mean of all input pixel values from each pixel value (so that the mean pixel value is now zero) [[2](#2)]
* max-norm regularization with a decaying learning rate [[3]](#3)
* `ReLu` vs. `tanh` as an activation function [[5]](#5)

When running the final models, we have experimented with applying more than a single model to each character image and then combining the results from the individual models into a single aggregate result. To combine multiple model results we take the geometric mean of the individual model results.

> <a id="1">1.</a> For example, see [Nielsen (2014) Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html).
> 
> <a id="2">2.</a> [Hinton et al. (2012) Improving neural networks by preventing co-adaptation of feature detectors](http://arxiv.org/pdf/1207.0580.pdf)
> 
> <a id="3">3.</a> [Srivastava et al. (2014) Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
>
> <a id="4">4.</a> Due initially to a coding mistake, we have found a benefit in dropping out the final logistic layer, as well as input or hidden layers. This seems to be a new application of dropout.
> 
> <a id="5">5.</a> Jarrett et al. (2009) What is the best multi-stage architecture for object recognition? In Proceedings of the International Conference on Computer Vision (ICCV’09). IEEE, 2009. (as cited in [[2](#2), [3](#3)])


### Slash categorization

What we're currently doing

What didn't work as well



### Digit categorization

What we're currently doing

What didn't work as well

What we'd like to investigate further

* bisector models
* different models for different digit positions

