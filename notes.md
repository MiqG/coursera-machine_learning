# Machine Learning (Andrew Ng, Stanford University)
Source: https://www.coursera.org/learn/machine-learning/home/welcome

## Introduction: Types of Machine Learning problems
### Supervised Learning
We try to learn from a **training dataset** that we assume is "correct" the parameters for best classification or regression. 
For that, the algorithm will take into account the combination of variables that are fed.
The way in which the features are mathematically processed depends on the kernel function of the ML algorithm.
### Unsupervised Learning
Conversely, if the dataset is unlabeled, we try to find structure within the data. For example, clustering algorithms.

## Linear Regression with One Variable
### Model and Cost Function
- Model

We represent the model with **m** to denote the number of training examples, **x** to denote the "input" variable/features and **y** to denote the "output"/"target" variable.
We feed the *training set* to a learning algorithm that generates a **hypothesis function** that relates the variables in the way we desire.

$$y=h(x_{training~set})$$

![](figures/fig_model_representation.png)

- Cost Function

The **cost function** is used to measure the accuracy of our hypothesis function during optimization of the parameters to training data.
For instance, if the hypothesis is linear -we want to fit a linear model to data-, the parameters will be slope and intercept.
Usually, we **minimize** the sum squared errors between the training examples and the function of the training examples.

$$J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x_i)-y_i)²$$

We can make a *contour plot* to represent the value of the cost function at each combination of parameters.
Lines of the same color correspond to the same cost function value but with different parameter combinations.
With two parameters, the optimal combination corresponds to the center of the circle.

![](figures/fig_contour_plot.png)

### Parameter learning

- Gradient descent

Machine learning algorithm to minimize an arbitrary function $$J(\theta_0,\theta_1)$$. 
We start with some combination of parameters and keep changing them while the function reduces.
We take little steps to go down the steepest slope until convergence.
With the limitations that finding the global minimum is not guaranteed, and that we will need to select a **learning rate or step size** (**$\alpha$**) and the number of **parameter combinations** that can be tested.

        - Algorithm:

Repeat until convergence:    
$$\theta_j := \theta_j - \alpha\frac{\delta}{\delta_j}J(\theta_0,...,\theta_n)~where~j=0,...,n_{parameters}$$ 
with simultaneous update until convergence.



Importantly, for each iteration, both parameters are updated simultaneously.

![](figures/fig_grad_descent.png)

The **learning rate** size can cause the gradient descent to be slow to converge if $\alpha$ is too small and diverge if $\alpha$ is too large.

Another issue is that depending on the number of **parameter combinations** that we initialize optimization at a large number of parameter combinations and compare the minimums achieved to be sure that we do not get stuck at an initial local minimum.

If a function is convex, it will always converge to the global minimum (e.g. linear regression).

Now we are considering a "batch" gradient descent where all variables are optimized simultaneously; we make this distinction because other optimization algorithms take sets of variables to optimize sequentially.

## Linear Regression with Multiple Variables
### Multivariate Linear Regression
We use **n** to denote the number of features and **m** the number of examples/observations/rows for each combination of features.
Now, we use matrix notation to represent linear combinations of n parameters with n features, for each observation.

#### Multivariate Gradient Descent
- Feature Scaling
Scaling the features makes gradient descent faster because changes in one feature may not contribute as much.
We can achieve scaling form 0 to 1 by dividing by the sum of all feature values.
In general, we scale by **mean normalizing** the data with mean 0 and boundary values -1 and 1:

$$x_i' = \frac{x_i-\mu_i}{std_i}$$

- Learning Rate:
To ensure that gradient descent works properly, the value of the cost function with the combination of parameters should descend at each iteration.
Plotting the values of the cost function at each iteration, the function reaches a plateau; we consider that we achieved convergence.

If the values of the cost functions increase or fluctuate with iterations, we usually should use a smaller $\alpha$.

Therefore, choosing the right $\alpha$ is completely empirical.

### Computing Parameters Analytically
- Feature transformation
Depending on the insight in the topic we can define transform the features and result into a better model.

For example, house prices with respect to prices usually rise very fast and reach a plateau.

Probably a quadratic function would fit the data well enough, but increasing the size for future predictions would lead to smaller prices, which does not make any sense.

Therefore, we could choose a different polynomial function to change the behavior of the curve generates by out hypothesis function.

Remember to scale features accordingly to the power applied on features.

- Normal Equation

To minimize the cost function for each combination of parameters, we perform the partial derivatives of each parameter and solve for 0 through the normal equation.
Alternatively to gradient descent, we can implement the normal equation to find the parameters that best fit the data without the need of feature scaling.
The main problem of this approach is that computing the inverse is computationally expensive if n is large compared to gradient descent.

$$\theta = (X^T X)^{-1}X^Ty$$

![](figures/fig_example_normal_eq.png)

As a rule of thumb, one should use the normal equation up to $10^4$ observations.

The non-invertibility of the normal equation may be an issue when implementing this approach.
There might be 2 reasons:

a. redundant features (linearly dependent): delete a variable. 
b. too many features (number of features exceeds the number of observations: delete variables or use regularization.


## Logistic Regression
### Classification and Representation
### Logistic Regression Model
### Multiclass Classification
## Regularization
### Solving the Problem of Overfitting
## Neural Networks
### Representation
### Cost Function and Backpropagation
## Advice for Applying Machine Learning
### Evaluating a Learning Algorithm
### Bias vs. Variance
## Machine Learning System Design
### Building a spam filter
### Handling Skewed Data
### Using Large Data Sets
## Support Vector Machines (SVM)
### Large Margin Classification
### Kernels
### SVMs in Practice
## Unsupervised Learning
### Clustering
## Dimensionality Reduction
### Principal Component Analysis
### Applying PCA
## Anomaly detection
### Density Estimation
### Building an Anomaly Detection System
### Multivariate Gaussian Distribution
## Large Scale Machine Learning
### Gradient Descent with Large Datasets
### Advanced topics
## Application Example: Photo OCR