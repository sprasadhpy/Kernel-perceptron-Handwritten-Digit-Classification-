# Kernel-perceptron-Handwritten-Digit-Classification-
Coursework part of the Supervised Learning Course(COMP0078)

# Generalisation Methods

## One-versus-One Classification

We first propose a one-versus-one method to generalize binary classification. Suppose we have \( k \) classes, then under this setting, we need to construct \( \frac{k(k - 1)}{2} \) classifiers, each trained using only the data from 2 of the \( k \) classes. Therefore, each classifier is responsible for distinguishing 2 classes of data. When it comes to prediction, we will employ a voting system: given an unseen data point, we feed it into every classifier and record the results. The class that has the most \( +1 \) predictions is the final output.

### One versus one generalised Kernel Perceptron

| Step | Description |
| ---- | ----------- |
| Input | \( (x_1, y_1), \ldots, (x_m, y_m) \in \mathbb{R}^n \times \{-1, 1\} \) |
|  | Compute Gram matrix \( K \) using the kernel function: \( K_{ij} = K(x_i, x_j) \) |
|  | Initialise \( w^{(p)} = 0 \) for \( p \in \{1, \ldots, k(k - 1)/2\} \) |
| 1. | For \( t \leq \text{max iteration} \) do |
| 2. | Receive data \( x_t, y_t \) |
| 3. | Calculate \( \text{sign}\left(\sum_i w_i^{(mn)} K(x_i, x_t)\right) \) and obtain the prediction of each classifier |
| 4. | Count the predictions and assign the class with the most vote as \( \hat{y_t} \) |
| 5. | For \( p \) from \( 1 \) to \( k(k - 1)/2 \) do |
| 6. | If the classifier involves \( y_t \) and the classifier made the wrong prediction then |
| 7. | \( w_t^{(p)} = w_t^{(p)} + 1 \) if \( y_t \) is the positive class |
| 8. | \( w_t^{(p)} = w_t^{(p)} - 1 \) if \( y_t \) is the negative class |
| 9. | End if convergence |

## One-versus-All Classification

We initially suggest the one versus-all approach to extend the binary classification algorithm. This method necessitates the creation of numerous binary classifiers, each tasked with distinguishing a single data class from all others. Consequently, we would possess several decision boundaries. Upon predicting with unseen data:

\[ \hat{y}_i = \underset{k}{\mathrm{arg\,max}} \, \mathbf{w}_k \cdot \mathbf{x}_i \]

The dot product may be regarded as a measure of confidence by which the data is categorized according to which weight engenders the highest confidence. These confidences can also be viewed as the distance from the boundary in a positive direction.

### One versus Rest generalised Kernel Perceptron

| Step | Description |
| ---- | ----------- |
| Input | \( (x_1, y_1), \ldots, (x_m, y_m) \in \mathbb{R}^n \times \{-1, 1\} \) |
|  | Compute Gram matrix \( K \) using the kernel function: \( K_{ij} = K(x_i, x_j) \) |
|  | Initialise \( w_1^{(k)} = 0 \) for all k |
| 1. | For \( t \leq \text{max iteration} \) do |
| 2. | Receive data \( x_t, y_t \) |
| 3. | Make prediction \( \hat{y}_t = \arg\max_k \sum_i w_i^{(k)} K(x_i, x_t) \) |
| 4. | Compute \( \hat{y}_t y_t \) |
| 5. | If \( \hat{y}_t y_t \leq 0 \): then |
| 6. | \( w_t^{(y_t)} = w_t^{(y_t)} + 1 \) |
| 7. | \( w_t^{(\hat{y}_t)} = w_t^{(\hat{y}_t)} - 1 \) |
| 8. | End if convergence |

## Preliminary Considerations

**Early Stopping**

An investigation into the learning trajectory or convergence pattern of the algorithm is conducted
