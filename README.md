## Data Science
The goal of this repository is to summarize and organize my knowledge for machine learning. All the codes will be written in Jupyter Notebook format, and should be reproducible by either cloning or downloading the whole repository. 

---
### Featured Projects

* **Zillow's Home Value Prediction** [[Repository]](https://github.com/patrick-ytchou/Kaggle-Zillow-Home-Value)

| Model | Private Leaderboard Score | Private Leaderboard Ranking | Percentile (Top) |
| :---: | :---:| :---: | :---: |
| Random Forest | 0.07540 | 760 / 3770 | 20.2% |
| CatBoost | 0.07514 | 250 / 3770 | 6.6% |
| **Stacking** | **0.07505** | **120 / 3770** | **3.2%** |


* **Restaurant Recommender System** [[Repository]](https://github.com/patrick-ytchou/Restaurant-Recommender-System)

***Project Introduction***

This is a recommender system using matrix factorization and deep learning-based residual learning methods hosted on AWS EC2 instances. It aims to provide a interface for user to input its preference to different kinds of movies. From their initial input, the model will generate recommendations accordingly.

* **Customer Review Generation** [[Repository]](https://github.com/patrick-ytchou/Customer-Review-Generation)

***Project Introduction***

This is aa text generation mdoel trained via OpenAI's GPT-2 language model. Transfer learning is implemented to fine-tune general GPT-2 model so that it can generate customer reviews better.

---
### Machine Learning Notes


#### **Special Topic**
* **Yelp Review Analysis: Sentiment Analysis & Topic Modeling** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/SpecialTopic/Yelp%20Review%20Analysis%20--%20Sentiment%20Analysis%20%26%20Topic%20Modeling.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/SpecialTopic/Yelp%20Review%20Analysis%20--%20Sentiment%20Analysis%20%26%20Topic%20Modeling.ipynb)
    * Sentiment Analysis (Textblob, VADER, Afinn)| Topic Modeling | Natural Language Processing 

* **Market Basket Analysis: Association Rule Mining** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/SpecialTopic/Market%20Basket%20Analysis%20--%20Association%20Rule%20Explained.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/SpecialTopic/Market%20Basket%20Analysis%20--%20Association%20Rule%20Explained.ipynb)
    * Association Rule Mining | Market Basket Analysis | Apriori Algorithm

* **Model Explanation with Santander Dataset** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/SpecialTopic/Model%20Explanation%20with%20Santander%20Dataset.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/SpecialTopic/Model%20Explanation%20with%20Santander%20Dataset.ipynb)
	* Tree Visualization | Permutation Feature Importance | Partial Dependence Plot | SHAP Values

* **PU Learning** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/SpecialTopic/PU%20Learning.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/SpecialTopic/PU%20Learning.ipynb)
    * PU Learning | Clustering | PU Bagging | Two-step Methods

#### **Classification**

* **Decision Tree Classifier From Scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Tree/Decision%20Tree%20Classifier%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Tree/Decision%20Tree%20Classifier%20from%20Scratch.ipynb)
    * Gini Impurity | Parent/Child Nodes | Decision Tree | Tree Visualization

* **K-Nearest Neighbors from Scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Classification/K-Nearest%20Neighbors%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Classification/K-Nearest%20Neighbors%20from%20Scratch.ipynb)
    * Minkowski | Regressor | Classifier | KD-Tree | Ball-Tree | Weighted KNN

* **Introduction to Logistic Regression from scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Classification/Introduction%20to%20Logistic%20Regression%20from%20scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Classification/Introduction%20to%20Logistic%20Regression%20from%20scratch.ipynb)
    * Logistic Regression | Threshold Analysis | Solvers | Logit | Log Odd

#### **Clustering**
* **Gaussian Mixture Model and EM Algorithm** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Clustering/Gaussian%20Mixture%20Model%20and%20EM%20Algorithm.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Clustering/Gaussian%20Mixture%20Model%20and%20EM%20Algorithm.ipynb)
    * Gaussian Mixture Model | Expectation Maximization

#### **Regression**

* **Regression Analysis: Assumptions for Linear Regression** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb)
    * Multicollinearity | Heteroscedasticity | Auto-Correlation
    * ResidualsPlot | White Test | Q-Q Plot | Durbin-Watson
    
* **Regression Analysis: Regression with Regularization** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb)
    * Naive Linear Regression | Regularization | Lasso/Ridege Regression
    
* **Regression Analysis: Parametric/Nonparametric regression** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb)
    * Polynomial Regression | Random Forest Regression | dtreeviz
    
* **Regression Analysis: Loss Functions for Regression Analysis** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb)
    * MSE | MAE | RMSE | MBE | MAPE | RMSLE | R² | Adjusted R²

#### **Neural Network**
* **Tips for Neural Network Training** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Tips%20for%20Neural%20Network%20Training.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Tips%20for%20Neural%20Network%20Training.ipynb)
	* Activation Function | Optimizer | EarlyStopping | Regularization | Dropout

* **Convolutional Neural Network (CNN) with MNIST** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Convolutional%20Neural%20Network%20with%20MNIST.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Convolutional%20Neural%20Network%20with%20MNIST.ipynb)
	* Convolution | Max Pooling | Flatten

* **Simple Recurrent Neural Net (RNN) from Scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Simple%20Recurrent%20Neural%20Net%20(RNN)%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Simple%20Recurrent%20Neural%20Net%20%28RNN%29%20from%20Scratch.ipynb)
	* Recurrent Neural Network | Vanishing Gradient Problem 

* **Time Sequence Prediction with LSTM** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Time%20Sequence%20Prediction%20with%20LSTM.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/NeuralNetwork/Time%20Sequence%20Prediction%20with%20LSTM.ipynb)
	* LSTM | Forget Gate | Input Gate | Output Gate 


#### **Optimization**

* **Genetic Algorithm from scratch: Traveling Salesman Problem** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb)
    * Genetic Algorithm | Traveling Salesman Problem 

* **Gradient Descent from scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/Gradient%20Descent%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Algorithms/Optimization/Gradient%20Descent%20from%20Scratch.ipynb)
	* Vanilla Gradient Descent | Adagrad | Stochastic Gradient Descent


#### **Analytics**

* **A/B Testing: Determine Sample Size** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Analytics/AB-Testing%20-%20Determine%20Sample%20Size.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Analytics/AB-Testing%20-%20Determine%20Sample%20Size.ipynb)
	* A/B Testing | Effect Size | Cohen's d

* **Cohort Analysis & RFM Model in Python** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Analytics/Cohort%20Analysis%20%26%20RFM%20Model%20in%20Python.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Analytics/Cohort%20Analysis%20%26%20RFM%20Model%20in%20Python.ipynb)
    * Time-based Cohort Analysis | HeatMap

* **BIGTARGET Uplift Modeling** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/Analytics/Marketing%20Campaign%20Analysis%20with%20Uplift%20Modeling.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/Analytics/Marketing%20Campaign%20Analysis%20with%20Uplift%20Modeling.ipynb)
    * Uplift Modeling | Causal Inference | Two-Models Modeling | Class Transformation 

#### **Feature Engineering**

* **Imbalanced Data Manipulation** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/FeatureEngineering/Imbalanced%20Dataset%20Manipulation.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/FeatureEngineering/Imbalanced%20Dataset%20Manipulation.ipynb)
    * Imbalanced Data Manipulation | Tomek's Link | SMOTE | SMOTETomek

---
### Resource

This section includes links for resources for future reference.

**Statistics**
[[Book] Distribution Explorer](https://distribution-explorer.github.io/)

**Machine Learning**
[[Book] Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)

[[Book] The Machine & Deep Learning Compendium](https://mlcompendium.gitbook.io/machine-and-deep-learning-compendium/)

[[Notes] Transformers from Scratch](https://e2eml.school/transformers.html)

[[Slides] Causal Inference in Marketing](https://www.slideshare.net/TAWEIHUANG/causal-inference-in-marketing?fbclid=IwAR0F5pZHaIAXdDdchhX5K3v_a0vc5AwFZuVS3yDXqAjcS3jjURMFlUvl7Q0)

**MLOps**
[[Courses] MLOps-Basics](https://github.com/graviraja/MLOps-Basics)
