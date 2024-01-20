# Kernel-perceptron-Handwritten-Digit-Classification-
Coursework part of the Supervised Learning Course(COMP0078)
Overview

This project details the implementation of a Kernel Perceptron algorithm for the classification of handwritten digits. We explore two generalization methods: One-versus-One Classification and One-versus-All Classification.

Generalisation Methods

One-versus-One Classification
In the one-versus-one method, we extend binary classification to multiclass scenarios. Given k classes, we create k(k - 1)/2 classifiers. Each classifier is trained on data from two of the k classes, distinguishing between these two classes.

Algorithm:

Input: Data points (x_1, y_1), ..., (x_m, y_m) in ℝ^n x {-1, 1}.
Initialisation: Compute the Gram matrix K with K_ij = K(x_i, x_j). Initialise w^(p) = 0 for each classifier p.
Procedure:
For each iteration t <= max iteration:
Receive data x_t, y_t.
Calculate the prediction for each classifier.
Determine the class with the most votes as y_hat_t.
For each classifier p:
Update weights if the classifier's prediction was incorrect.


One-versus-All Classification
In the one-versus-all approach, we create binary classifiers to distinguish each class from all others. This leads to multiple decision boundaries for classification.

Prediction:

Given unseen data, the predicted class y_hat_i is determined by:

y_hat_i = arg max_k (w_k . x_i)
Here, the dot product represents a confidence level, indicating the class with the highest confidence for each data point.

