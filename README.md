# Data Science
The goal of this repository is to summarize and organize my knowledge for machine learning. All the codes will be written in Jupyter Notebook format, and should be reproducible by either cloning or downloading the whole repository. The content of the notebook aims to strike a good balance between background knowledge, mathematical formulations, naive algorithm implementation (via numpy, pandas, statsmodels, scipy, matplotlib, seaborn, etc.), and sophisticated implementation through open-source library.

---
## Featured Projects

* **Zillow's Home Value Prediction** [[Directory]](https://github.com/patrick-ytchou/Data-Science/tree/master/Projects/ZillowHomeValue)

| Model | Private Leaderboard Score | Private Leaderboard Ranking | Percentile (Top) |
| :---: | :---:| :---: | :---: |
| LightGBM | 0.07540 | 760 / 3770 | 20.2% |
| CatBoost | 0.07514 | 250 / 3770 | 6.6% |
| **Stacking** | **0.07505** | **120 / 3770** | **3.2%** |


* **Yelp Review Analysis: Sentiment Analysis & Topic Modeling** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Projects/YelpReviewAnalysis/Yelp%20Review%20Analysis%20--%20Sentiment%20Analysis%20%26%20Topic%20Modeling.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Projects/YelpReviewAnalysis/Yelp%20Review%20Analysis%20--%20Sentiment%20Analysis%20%26%20Topic%20Modeling.ipynb)
    * Sentiment Analysis (Textblob, VADER, Afinn)| Topic Modeling | Natural Language Processing 

---
## Learning Notes

**Speical Topics**

* **Market Basket Analysis: Association Rule Mining** [[Notebook]](hhttps://github.com/patrick-ytchou/Data-Science/blob/master/Notes/AssociationRules/Market%20Basket%20Analysis%20--%20Association%20Rule%20Explained.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/AssociationRules/Market%20Basket%20Analysis%20--%20Association%20Rule%20Explained.ipynb)
    * Market Basket Analysis | Apriori Algorithm


**Machine Learning Algorithms**

* **Decision Tree Classifier From Scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Tree/Decision%20Tree%20Classifier%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Tree/Decision%20Tree%20Classifier%20from%20Scratch.ipynb)
    * Gini Impurity | Parent/Child Nodes | Decision Tree | Tree Visualization

* **K-Nearest Neighbors from Scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Clustering/K-Nearest%20Neighbors%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Clustering/K-Nearest%20Neighbors%20from%20Scratch.ipynb)
    * Minkowski | Regressor | Classifier | KD-Tree | Ball-Tree | Weighted KNN
  
* **PU Learning** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/PU%20Learning.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/PU%20Learning.ipynb)
    * PU Learning | Clustering | PU Bagging | Two-step Methods


**Linear Regression**
* **Regression Analysis: Assumptions for Linear Regression** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb)
    * Multicollinearity | Heteroscedasticity | Auto-Correlation
    * ResidualsPlot | White Test | Q-Q Plot | Durbin-Watson
    
* **Regression Analysis: Regression with Regularization** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb)
    * Naive Linear Regression | Regularization | Lasso/Ridege Regression
    
* **Regression Analysis: Parametric/Nonparametric regression** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb)
    * Polynomial Regression | Random Forest Regression | dtreeviz
    
* **Regression Analysis: Loss Functions for Regression Analysis** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb)
    * MSE | MAE | RMSE | MBE | MAPE | RMSLE | R² | Adjusted R²

**Neural Network**
* **Tips for Neural Network Training** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/NeuralNetwork/Neural%20Network%20-%20Tips%20for%20Neural%20Network%20Training.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/NeuralNetwork/Neural%20Network%20-%20Tips%20for%20Neural%20Network%20Training.ipynb)
	* Activation Function | Optimizer | EarlyStopping | Regularization | Dropout

* **Convolutional Neural Network (CNN) with MNIST** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/NeuralNetwork/Neural%20Network%20-%20Convolutional%20Neural%20Network%20with%20MNIST.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/NeuralNetwork/Neural%20Network%20-%20Convolutional%20Neural%20Network%20with%20MNIST.ipynb)
	* Convolution | Max Pooling | Flatten

**Optimization Techniques**

* **Genetic Algorithm from scratch: Traveling Salesman Problem** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb)
    * Genetic Algorithm | Traveling Salesman Problem 

* **Gradient Descent from scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Optimization/Gradient%20Descent%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ML%20Algos/Optimization/Gradient%20Descent%20from%20Scratch.ipynb)
	* Vanilla Gradient Descent | Adagrad | Stochastic Gradient Descent

**Model Explainability**

* **Model Explanation with Santander Dataset** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ModelExplanation/Model%20Explanation%20with%20Santander%20Dataset.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ModelExplanation/Model%20Explanation%20with%20Santander%20Dataset.ipynb)
	* Tree Visualization | Permutation Feature Importance | Partial Dependence Plot | SHAP Values

**Analytics**

* **A/B Testing: Determine Sample Size** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/AB%20Testing/AB-Testing%20-%20Determine%20Sample%20Size.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/AB%20Testing/AB-Testing%20-%20Determine%20Sample%20Size.ipynb)
	* A/B Testing | Effect Size | Cohen's d

**Feature Engineering**

* **Imbalanced Data Manipulation** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Notes/ImbalancedDataManipulation/Imbalanced%20Dataset%20Manipulation.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Notes/ImbalancedDataManipulation/Imbalanced%20Dataset%20Manipulation.ipynb)
    * Imbalanced Data Manipulation | Tomek's Link | SMOTE | SMOTETomek


---
## Tableau Visualization
Data Visualization is a critical skill not only for data exploration but also for the decision making processes, and Tableau is one of the most widely used software for visualization. As such, starting from April 2020 I deicde to put aside one hour weekly to create hone my visualization skills and find unique insights from data.

[Workout Wednesday](http://www.workout-wednesday.com/) is a weekly data visualization challenge to help anyone interested in data visualization to build on the skills in Tableau. Each Wednesday a challenge is release and participants are asked to replicate the challenge that is posed as closely as possible. Thanks to this wonderful community, there are numerous resources that we can learn visualization from.

Check out my [Tableau Public Gallery](https://public.tableau.com/profile/yung.tang.chou#!/) if you are interesed in more informative visualization!


---
## Pending Topics

* `Recommender System`

* `PU Learning`

* `Entity Embedding`