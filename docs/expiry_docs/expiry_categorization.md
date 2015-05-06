Expiry Categorization
---------------------

## Overview

In the field of computer vision, *segmentation* refers to isolating relevant elements from an overall image, while *categorization* refers to examining a particular element and determining what it represents.

As described in [Expiry Segmentation](expiry_segmentation.md), card.io's segmentation step for expiry isolates individual characters that are likely to belong to an expiration date. Probable slash characters (`/`) are categorized as part of the segmentation step.

Each of the remaining characters is then passed through the main expiry categorization step (described next) to classify it as a particular digit.

Finally, a few tests are applied to the resulting candidate month and year:

* Month must be in the range `01` through `12`.
* Month/Year must be in the range `now` through `5 years from now`. (The choice of 5 years is somewhat arbitrary. The more constrained it is, the fewer false positives. Though there are some cards with extraordinarily long expirations--most forms on the web appear to allow for 15 years!--the vast majority of cards expire within a few years, so we optimize for them.)
* Ignore a date if it is no later than the best date we've already found. (Some cards include start dates and issued-at dates as well as expiration dates.)


### Deep Learning models

Categorizing expiry digits is challenging! The characters are small, measuring only about 9x15 pixels at standard camera resolution. (See [Camera resolution](../camera_resolution.md).) Card backgrounds vary widely, as do the effects of various lighting conditions.

We use deep-learning neural-net models to perform categorization. The input to a model is a likely character image. The output is a set of category probabilities which sum to 1. For the slash categorizer, there are two categories: slash or non-slash. For the digit categorizer there are 10 categories: one for each digit.

We create our models using [Theano](http://www.deeplearning.net/software/theano/), combined with a library of Python utilities that we have created for our own use. For expiry categorization, we use [multilayer perceptron](http://www.deeplearning.net/tutorial/mlp.html#mlp) ("MLP") and [convolutional](http://www.deeplearning.net/tutorial/lenet.html#lenet) models. We began with the sample Theano code, but have since added a number of refinements. We then use code generation to convert our models (data) into executable C/C++ and commit the generated code.

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

We use a single model, a multilayer perceptron:

* input layer: 16 x 11 pixels
* hidden layer: 80 hidden units
* logistic layer: 2 outputs (is-not-slash, is-slash)

The model was trained on:

* 11,905 images of slashes (`/`)
* 79,117 images of "digits" (mostly actual digits, but some non-digits, and many miscategorized digits)
* 20,000 images of other character candidates (letters, parts of logos, bits of card background)

Before input, the images are processed with the x-axis Sobel operator described in [Expiry Segmentation](expiry_segmentation.md). 

Other approaches that failed to clearly perform better than this model included:

* Rather than x-axis Sobel, do no image processing.
* Rather than x-axis Sobel, process images with other filters.
* Convolutional models rather than MLP models.
* Different numbers of hidden units.


### Digit categorization

We use a single model, a multilayer convolutional model:

* input layer: 16 x 11 pixels
* convolutional layer #1: 50 kernels, each 5x5, with 2x2 max-pooling
* convolutional layer #2: 40 kernels, each 5x5, with 2x3 max-pooling
* full-connected layer: 176 hidden units
* logistic layer: 10 outputs (one for each possible digit)

The model was trained on:

* 12,767 images of slashes
* 50,084 images of digits (manually cleaned and categorized, but very heavily skewed toward `0`s and `1`s)
* 20,353 images of other character candidates

Before input, the images are processed with a sequence of computer-vision filters (gradient, equalization, bilateralization). This produces somewhat better results, with a small performance penalty, compared to using the x-axis Sobel processed image we already have on hand.

We experimented with hundreds of other parameter values, such as the number and size of convolutional and hidden layers, the types of image processing used, etc. We also experimented with combining the results from a few different convolutional and MLP models, but in the end found that the present single model produced equally good results.

### Some ideas that we'd like to investigate further

#### Different models for different digit positions

The first digit of a month can be only `0` or `1`. So rather than use a general-purpose digit categorizer, we should be able to create a smaller, faster, and more accurate categorizer that simply distinguishes between those two categories.

Similarly, if the first month digit is most likely a `1`, then that limits the second month digit to `0`, `1`, or `2`. But if the first month digit is most likely a `0`, then the second month digit cannot also be a `0`.

Other rules or heuristics can be applied to the digits of the year.

So there might be room for improvement by using different categorization models for each character position.

Unfortunately, each additional model will significantly increase the total size of the card.io library.


#### Bisector models

We have observed that MLP models run at least 30 times faster than convolutional models.

Therefore we could see a big performance gain if we could replace our current single convolutional model with a set of several MLP models. (Although this would increase our binary size footprint.)

One specific idea is as follows:

* Create a small set of models, each of which categorizes the ten digits into two roughly equal-sized sets.
  * For example, one model might distinguish "round" digits `{0, 2, 3, 6, 8, 9}` from "straight" digits `{1, 4, 5, 7}`.
  * Each other model would yield a different bisection.

* The bisections are chosen so that any pattern of bisection categorizations matches only a single digit.
  * For example, consider 4 carefully chosen categorizers.
  * Each bisection can be interpreted as categorizing into `category 0` vs. `category 1`.
  * The combined results from the four models can then be represented as a 4-bit pattern.
  * The bisections are chosen so that each of the 10 digits is represented by a unique 4-bit pattern.

In our initial experiments using our collection of card images, this approach showed promise. However, subsequent experiments on a phone with new cards proved disappointing.

(Note that each model could partition the ten digits into more than 2 sets. E.g., if each model categorized its inputs into 3 sets rather than 2, then the results from 3 models could be represented as a pattern of 3 base-3 digits.)
