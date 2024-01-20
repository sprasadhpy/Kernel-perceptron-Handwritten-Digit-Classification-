# Kernel-perceptron-Handwritten-Digit-Classification-
Coursework part of the Supervised Learning Course(COMP0078)

# Generalisation Methods

## One-versus-One Classification

We first propose a one-versus-one method to generalize binary classification. Suppose we have `k` classes, then under this setting, we need to construct `(k * (k - 1)) / 2` classifiers, each trained using only the data from 2 of the `k` classes. Therefore, each classifier is responsible for distinguishing 2 classes of data. When it comes to prediction, we will employ a voting system: given an unseen data point, we feed it into every classifier and record the results. The class that has the most `+1` predictions is the final output.

### One versus one generalised Kernel Perceptron

| Step | Description |
| ---- | ----------- |
| **Input** | `(x_1, y_1), ..., (x_m, y_m) in R^n x {-1, 1}` |
|  | Compute Gram matrix `K` using the kernel function: `K_ij = K(x_i, x_j)` |
|  | Initialise `w^(p) = 0` for `p in {1, ..., k * (k - 1) / 2}` |
| 1. | For `t <= max iteration` do |
| 2. | Receive data `x_t, y_t` |
| 3. | Calculate `sign(sum_i w_i^(mn) K(x_i, x_t))` and obtain the prediction of each classifier |
| 4. | Count the predictions and assign the class with the most vote as `y_hat_t` |
| 5. | For `p` from `1` to `k * (k - 1) / 2` do |
| 6. | If the classifier involves `y_t` and the classifier made the wrong prediction then |
| 7. | `w_t^(p) = w_t^(p) + 1` if `y_t` is the positive class |
| 8. | `w_t^(p) = w_t^(p) - 1` if `y_t` is the negative class |
| 9. | End if convergence |

## One-versus-All Classification

We initially suggest the one versus-all approach to extend the binary classification algorithm. This method necessitates the creation of numerous binary classifiers, each tasked with distinguishing a single data class from all others. Consequently, we would possess several decision boundaries. Upon predicting with unseen data:

y_hat_i = argmax_k (w_k * x_i)

The dot product may be regarded as a measure of confidence by which the data is categorized according to which weight engenders the highest confidence. These confidences can also be viewed as the distance from the boundary in a positive direction.

### One versus Rest generalised Kernel Perceptron

| Step | Description |
| ---- | ----------- |
| **Input** | `(x_1, y_1), ..., (x_m, y_m) in R^n x {-1, 1}` |
|  | Compute Gram matrix `K` using the kernel function: `K_ij = K(x_i, x_j)` |
|  | Initialise `w_1^(k) = 0` for all `k` |
| 1. | For `t <= max iteration` do |
| 2. | Receive data `x_t, y_t` |
| 3. | Make prediction `y_hat_t = argmax_k sum_i w_i^(k) K(x_i, x_t)` |
| 4. | Compute `y_hat_t * y_t` |
| 5. | If `y_hat_t * y_t <= 0`: then |
| 6. | `w_t^(y_t) = w_t^(y_t) + 1` |
| 7. | `w_t^(y_hat_t) = w_t^(y_hat_t) - 1` |
| 8. | End if convergence |

## Preliminary Considerations

**Early Stopping**

An investigation into the learning trajectory or convergence pattern of the algorithm is conducted to establish an appropriate early termination scheme. Experiments are carried out utilizing a polynomial kernel with degrees 3 and 5, respectively. These two models are trained on 80% of the dataset, with their generalization error assessed on the remaining 20%.



![image](https://github.com/sprasadhpy/Kernel-perceptron-Handwritten-Digit-Classification-/assets/40602129/3d3c57dd-e661-4a47-83f7-3721611ba413)



This indicates that a satisfactory outcome is achievable after approximately 5 epochs, thus setting the minimum number of iterations to 5.

## Learning Rate

Rather than incrementing the weight vector, **w**, by a unit at each iteration, a learning rate, **γ**, can be introduced to modulate the update magnitude. Empirical analyses reveal that adopting a diminished learning rate does not facilitate an expedited learning pace but serves to refine the smoothness of the learning trajectory.



## Model Evaluation Using Polynomial Kernels

To assess the performance of the algorithm with polynomial kernels of various degrees, seven unique models were developed, each corresponding to a different degree of the polynomial kernel. We implemented a random train-test split of 80% and 20% over twenty iterations for each model. The resulting training and testing errors were computed as mean values across these iterations, in line with the early stopping criterion established previously.

In general, increasing the polynomial degree tends to raise the model's complexity, which can potentially enhance accuracy. However, our analysis did not reveal any clear evidence of overfitting, even as the degree of the polynomial increased. Notably, the polynomial kernel with a degree of one exhibited the highest average error during both training and testing, succeeded by the kernel with a degree of two. The performances of higher degrees were less distinguishable, suggesting that they are all viable options for training a kernel perceptron.

### Mean and Standard Deviation of Training and Test Error Rate Over 20 Runs with Different Polynomial Kernel Degree in Percentage

| Degree | Training Error (Mean ± SD) | Testing Error (Mean ± SD) |
| ------ | --------------------------- | -------------------------- |
| 1      | 6.239 ± 0.311               | 8.384 ± 1.357              |
| 2      | 0.479 ± 0.107               | 3.809 ± 0.447              |
| 3      | 0.186 ± 0.114               | 3.099 ± 0.317              |
| 4      | 0.086 ± 0.057               | 3.035 ± 0.395              |
| 5      | 0.077 ± 0.063               | 3.269 ± 0.341              |
| 6      | 0.047 ± 0.032               | 3.323 ± 0.414              |
| 7      | 0.066 ± 0.036               | 3.379 ± 0.740              |


###   Train Error Vs. Test Error for different polynomial kernel degrees.
![image](https://github.com/sprasadhpy/Kernel-perceptron-Handwritten-Digit-Classification-/assets/40602129/7dee497e-b9c8-492a-b0b6-d93347ed62e1)

## Optimal Polynomial Degree Selection

To ascertain the optimal polynomial degree for the model, a series of twenty iterations of model selection were executed. Each iteration comprised a five-fold cross-validation process for each degree, facilitating the identification of the most suitable polynomial degree. Subsequently, models were retrained utilizing the selected degree on the holdout test set to compute the test error. The outcomes, presented in percentages, are explained as follows:

#Test Error Rates for optimal degrees
## Optimal Degree and Test Error Rate

| Optimal degree d* | Test Error Rate (%) |
| ----------------- | ------------------- |
| 5                 | 3.33                |
| 4                 | 3.98                |
| 5                 | 3.33                |
| 4                 | 3.17                |
| 6                 | 3.76                |
| 5                 | 3.28                |
| 7                 | 3.71                |
| 7                 | 3.66                |
| 4                 | 4.57                |
| 6                 | 3.33                |
| 5                 | 3.23                |
| 6                 | 3.49                |
| 5                 | 3.49                |
| 5                 | 2.74                |
| 7                 | 3.33                |
| 4                 | 3.49                |
| 6                 | 3.28                |
| 7                 | 2.31                |
| 6                 | 2.74                |
| 5                 | 3.60                |

## Summary of Model Evaluation

Average Degree: 5.45, Standard Deviation: 1.02  
Average Test Error Rate (%): 3.39, Standard Deviation: 0.46

The calculated mean optimal polynomial degree is 5.45 with a standard deviation of 1.02, alongside a mean test error rate of 3.39% and a standard deviation of 0.46. Given that the median of the recorded degrees is also 5 across these twenty evaluations, this degree should be the preferred choice for polynomial complexity in future model training.

## Evaluation of Kernel Perceptron using Confusion Matrix

An alternative method for evaluating the kernel perceptron's performance involves the utilization of a confusion matrix. This analytical tool elucidates the nature of classification errors made by the algorithm and identifies class pairs that are particularly challenging to discriminate. The methodology remains consistent with previous practices, involving cross-validation to determine the optimal polynomial degree. Following this, the model is retrained, and a confusion matrix is computed, with its entries being averaged across twenty iterations.

### Confusion Matrix of Optimal Polynomial Degree (Average over 20 Runs)

|      | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **0** | 0.0   | 0.0   | 0.15  | 0.115 | 0.151 | 0.174 | 0.234 | 0.025 | 0.134 | 0.017 |
| **1** | 0.0   | 0.0   | 0.043 | 0.083 | 0.361 | 0.0   | 0.22  | 0.103 | 0.08  | 0.01  |
| **2** | 0.159 | 0.04  | 0.0   | 0.153 | 0.246 | 0.043 | 0.026 | 0.182 | 0.139 | 0.012 |
| ...  | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   |
| **9** | 0.014 | 0.02  | 0.048 | 0.094 | 0.495 | 0.013 | 0.003 | 0.236 | 0.027 | 0.0   |

*Note: Table truncated for brevity.*

### Standard Deviation of the Confusion Matrix (Average over 20 Runs)

|      | 0     | 1     | ... | 9     |
|------|-------|-------|-----|-------|
| **0** | 0.0   | 0.0   | ... | 0.05  |
| **1** | 0.0   | 0.0   | ... | 0.037 |
| ...  | ...   | ...   | ... | ...   |
| **9** | 0.062 | 0.073 | ... | 0.0   |

*Note: Table truncated for brevity.*

The analysis results are markedly interpretable. The model demonstrates notable challenges in differentiating between digit classes that exhibit similar morphological characteristics, exemplified by pairs such as 3 and 5, 4 and 9, 7 and 9, and 0 and 6. Conversely, digit classes with distinct morphological distinctions are classified with a higher degree of accuracy, evidenced by the absence of misclassifications among these discernibly different pairs.

## Most Often Misclassified Data

We investigate further when the algorithm makes a mistake. This time the procedure consists of 40 runs, for each run the model is trained using 80% of the data and evaluated using the entire dataset. Data that are misclassified are recorded during each run. In the end, the 4 data points which are most often misclassified are displayed down below:

![image](https://github.com/sprasadhpy/Kernel-perceptron-Handwritten-Digit-Classification-/assets/40602129/0fee1a4a-7098-48d6-8ee2-dabd1662d664)


The misclassification of the last two '1' digits is attributed to their incorrect labelling or atypical representation, with both being erroneously labelled as '4'. Furthermore, the '8' digits are characterised by untidy handwriting, rendering them challenging to decipher, even for human observers. Consequently, it is rational to anticipate that while our model attains high accuracy, it does not reach a perfect 100% accuracy rate. This expectation is underpinned by the likelihood of encountering more instances of similarly ambiguous data.

## Employing the Gaussian Kernel in the Model

As an alternative, the Gaussian kernel can be employed as the kernel function in our model. A notable benefit of this kernel is its capacity to project inputs into an infinitely dimensional feature space. This capability allows the algorithm to exhibit high complexity while maintaining computational efficiency. The Gaussian kernel function is defined as:

K(p, q) = exp(-c ||p - q||^2)


Here, `c` corresponds to `1 / (2σ^2)`, regulating the influence a data point exerts on the prediction based on its distance from the point of interest. A larger value of `c` implies that data points located further away will have comparatively less influence.

In our investigation of the suitable range for the hyperparameter `c`, we began with a series of experiments covering `c` values ranging from `100` to `10^(-7)`. It became evident that when `c` dropped below `10^(-4)`, there was a significant increase in the test error rate, indicating challenges in achieving robust generalization. This phenomenon appeared to be linked to distant data points playing a minimal role in influencing the model's performance. As a result, we refined the acceptable range to `{3^(-1), ..., 3^(-7)}` and repeated the experiments, providing reports on both training and testing error rates.

### Observations on Gaussian Kernel's Performance

In the initial stages, `c` was set at a relatively high value, resulting in a slightly elevated generalization error. This could be attributed to data points located farther from the decision boundary having limited impact. As we reduced `c` to `3^(-4)`, we observed a corresponding decrease in generalization error. Further reductions in `c` values revealed that distant data points exerted excessive influence, leading to biased predictions, a behavior analogous to what we encountered in our first coursework involving the k-NN algorithm.

### Mean and Standard Deviation of Training and Test Error Rates with Different `c` Values

| `c`      | Train Error (Mean ± SD) | Test Error (Mean ± SD) |
|----------|-------------------------|------------------------|
| `3^(-1)` | 0.000 ± 0.000           | 5.989 ± 0.553          |
| `3^(-2)` | 0.009 ± 0.015           | 5.091 ± 0.562          |
| `3^(-3)` | 0.040 ± 0.033           | 3.449 ± 0.459          |
| `3^(-4)` | 0.203 ± 0.095           | 3.145 ± 0.355          |
| `3^(-5)` | 1.250 ± 0.134           | 4.468 ± 1.083          |
| `3^(-6)` | 5.501 ± 0.259           | 7.444 ± 1.645          |
| `3^(-7)` | 9.573 ± 0.361           | 9.562 ± 1.874          |






