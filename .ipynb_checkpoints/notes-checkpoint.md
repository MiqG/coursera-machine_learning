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
ML in classification problems try discriminate between combinations of features.
For example, we could use linear regression to find a threshold between values.
But, there are approaches that can better label.

### Logistic Regression Model
Even though it is called regression, it is a classification algorithm.
- Hypothesis Representation
We want our classifier to give us values between 0 and 1.
Then, we can define the hypothesis function as a **sigmoid or logistic function**:

$$h_\theta(x)=\frac{1}{1+\exp^{-\theta^Tx}}$$

![](figures/fig_sigmoid_func.png)

Which looks like a switch that is asymptotic at y=0 for $-\infty$ and y=1 for $+\infty$.
Now, we interpret the hypothesis function as a probability of classification:

$$h_\theta(x)=P(y=1|x;\theta)$$

- Decision Boundary

The hypothesis outputs the estimated probability of data being classified as 1 or 0.
The sigmoid function per se is greater or equal to 0.5 when $\theta^Tx\geq0$.
This hypothesis function will create a **decision boundary** between the two groups; it is a property of the parameters calculated from the dataset.
If the dataset has no linearity properties, we can add features by polynomial transformations to get more complex decision boundaries.

- Cost Function

We have a training set with m examples.
Now, we use an alternative cost function that compares the predicted with the actual value.
If we would use the same cost function than in linear regression, since our hypothesis function is not convex anymore, we would run into a non-convex problem hindering finding a global minimum through gradient descent.

$$
Cost(h_\theta(x),y)=
\begin{cases}
      -log(h_\theta(x)), & \text{if}\ y=1 \\
      -log(1-h_\theta(x)), & \text{if}\ y=0
\end{cases}
$$
![](figures/fig_sigmoid_cost1.png)
![](figures/fig_sigmoid_cost2.png)

- Simplified cost function and gradient descent

We can simplify how the cost function to ease its computation:

$$Cost(h_\theta(x),y)=-\frac{1}{m}[\sum_{i=1}^{m}y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$$

We will perform gradient descent to find the parameters that minimize the values of the cost function.
Even though the hypothesis function has changed, we perform the same parameters update function.

- Advanced Optimization

We need to compute the cost function and its partial derivatives for a number of iterations.
A part from Gradient Descent, we could use:

    - Conjugate descent: https://en.wikipedia.org/wiki/Conjugate_gradient_method
    - BFGS (Broyden–Fletcher–Goldfarb–Shanno): https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm
    - L-BFGS: https://en.wikipedia.org/wiki/Limited-memory_BFGS

We are not going to cover them deeply.
Their main advantages are that they do not need to pick $\alpha$ manually and that they usually converge faster than gradient descent.
However, they are quite more complex to implement.



### Multiclass Classification

- One vs all method

For example, we could automatically tag our emails in multiple classes, or perform multiple diagnosis of patients.
We want to implement the same principles of binary classification problems into multiple.
If we want to split the data into 3 groups, we would perform 3 binary classifications that will result in 3 fitted classifiers, each of them trained to recognize each class.
Finally, we pick the classifier that retrieves the highest probability of being that value.

![](figures/fig_multiclass_onevsall.png)

## Regularization
### Solving the Problem of Overfitting
When a model does not fit data very well, we consider that it **underfits** data introducing a strong **bias** mathematically to make accurate predictions.
**Overfitting** happens when we fit the model to data too well and it performs bad in generalizing the model to test data; unseen examples.

With low-dimensional data, we can easily visualize it.
Conversely, to prevent overfitting with high-dimensional data and low number of observations

- Reduce number of features: 
    - manually
    - feature selection algorithms.
- Regularization:
    - keep all features, but reduce magnitude/values of parameters $\theta_j$.
    - works well when lots of features that contribute a bit in predicting $y$.
![](figures/fig_overfit.png)

### Regularization through cost function

We could introduce a penalty within the cost function making parameters small that results in a "simpler" hypothesis function less prone to overfitting.

Since we do not know which parameters we should shrink, we shrink all of them summing a regularization term that contains a **regularization parameter** ($\lambda$).

- Regularized linear regression

$$
J(\theta_0)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})²+\lambda\sum_{j=1}^{n}\theta_j^2
$$

If $\lambda$ is very large we would end up underfitting.

In the parameter update, we only apply regularization from j=1 to j=n; not j=0.
This penalty results in a sum term that will decrease the value of the parameter.

$$\theta_0=[\theta_0-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)}]for~j=0$$

$$\theta_j=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{n}\theta_j
=[\theta_j(1-\alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}]_{for~j=1,2,...,n}$$


If we perform the **normal equation** method, parameters can be computed through:

$$\theta = (X^TX+\lambda diag(0,1,1,...,1))^{-1}X^Ty$$

Recall that if m < n, then $X^TX$ is non-invertible. However, when we add the regularization term, then it becomes invertible.

- Regularization for Logistic Regression

We simply add a regularization term based on the parameters to the cost function:

$$Cost(h_\theta(x),y)=-\frac{1}{m}[\sum_{i=1}^{m}y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{1}{2m}\sum_{j=1}^{n}\theta_j^{2}$$

And we perform exactly the same update of the parameters.

## Neural Networks
### Representation

- Non-linear hypotheses
- Neurons and the Brain
- Neural Networks
- Applications

### Learning

- Cost Function and Backpropagation
- Backpropagation in Practice
- Application

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