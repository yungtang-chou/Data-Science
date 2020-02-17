# Data Science

As I start my journey in data science, I will continuously update this repository to document my learning. This repository will be divided into the following sections: featured projects, data science learning notes, and other special topics.

The goal of this repository is to summarize and organize my knowledge for machine learning. All the codes will be written in Jupyter Notebook format, and should be reproducible by either cloning or downloading the whole repository. The content of the notebook aims to strike a good balance between background knowledge, mathematical formulations, naive algorithm implementation (via numpy, pandas, statsmodels, scipy, matplotlib, seaborn, etc.), and sophisticated implementation through open-source library.

---
## General Framework

Credit to the wonderful post [Don’t Do Data Science, Solve Business Problems](https://towardsdatascience.com/dont-do-data-science-solve-business-problems-6b70c4ee0083) by [Cameron Warren
](https://towardsdatascience.com/@camwarrenm), as well as the post[Solve Business Problems with Data Science](https://medium.com/@jameschen_78678/solve-business-problems-with-data-science-155534b1995d) by [James Chen](https://medium.com/@jameschen_78678),  it's important not to learn how to implement machine learning models, but to solve real business problems. 

In this section, in each project I will follow the general framework below:

1. **`Define Business Problems`**

The main goal here is to clearly define what the busines problems are. If there's no clear business value or strategic goal, all the work will be simply futile. Defining the main goal in advance also helps limiting the scope of the problem, making it possible to solve a complicated business problem piece by piece. Only after clarifying the business problem can we understand what to do next.

2. **`Outline Analytics Objectives`**

There are plenty of algorithms out there that might suit your problem, or probably only a limited amount of them fit in. It's data scientist's responsibility to define the objective of this problem and which model to implement. Is supervised or unsupervised more applicable? Is this a regression problem or a classification one? How should I define the target metric if there's no clear one?

Step 1 and 2 can be an iterativd process, as the initial business problem might be infeasible or not well-defined. Only after the problem and object are well-defined can we start the modeling process.


3. **`Prepare Data for Modeling`**

This process includes **Data Collection**, **Data cleansing** and **Feature Engineering**.

In the real world data collection is an important topic, especially for those companies that are not internet-based. In this repository, most of the data are collected via open data sources and will be pointed out to maintain reproducibility. I will also try to discuss some idea about how to collect data in practice.

Data cleansing and feature enginnering are known to be the processes that require the most amount of time, especially in the real world. I will try to go through these complex process via concise code with reasoning.


4. **`Develop Machine Learning Model`**

In this section I will compile the model to solve this problem. The complexity of the model depends on whether the main idea of this topic is about the model itself. If not, simple and quick benchmark model will be implemented and will likely not be optimized.


5. **`Evaluation and Notes for Future Improvement`**

In this last section, I will try to evaluate our result based on the business problem defined in the first step. Does our solution really solve the problem at hand? What are some other potential improvements that can work better? Different modeling technique? Another analytics objectives?



---
## Featured Projects




---
## Data Science Learning Notes

**Regression Analysis**
* 2020-02-11 **`Regression Analysis: Assumptions for Linear Regression`** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb)
    * Multicollinearity | Heteroscedasticity | Auto-Correlation
    * ResidualsPlot | White Test | Q-Q Plot | Durbin-Watson
    
* 2020-02-11 **`Regression Analysis: Regression with Regularization`** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb)
    * Naive Linear Regression | Regularization | Lasso/Ridege Regression
    
* 2020-02-11 **`Regression Analysis: Parametric/Nonparametric regression`** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb)
    * Polynomial Regression | Random Forest Regression | dtreeviz
    
* 2020-02-16 **`Loss Functions for Regression Analysis`** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb)
    * MSE | MAE | RMSE | MBE | MAPE | RMSLE | R² | Adjusted R²



**Classification Analysis**



**Clustering Analysis**



**Neural Network**



**Optimization Techniques**

* 2020-02-15 **`Genetic Algorithm from scratch: Traveling Salesman Problem`** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb)
    * Genetic Algorithm | Traveling Salesman Problem 
<p align="center">
	<img src="https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/TSP_animation.gif" width=300 height=200/>
</p>


**Model Selection & Explainability**




**Feature Engineering Technique**



---
## Pending Topics

* `Natural Language Processing Project (Sentiment Analysis + Topic Modeling)`

* `Imbalanced Data Manipulation`

* `Clustering Analysis Series`

* `Gradient Descent From Scratch`

* `Recommender System`
