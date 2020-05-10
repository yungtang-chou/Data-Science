# Data Science

As I start my journey in data science, I will continuously update this repository to document my learning. This repository will be divided into the following sections: featured projects, data science learning notes, and other special topics.

The goal of this repository is to summarize and organize my knowledge for machine learning. All the codes will be written in Jupyter Notebook format, and should be reproducible by either cloning or downloading the whole repository. The content of the notebook aims to strike a good balance between background knowledge, mathematical formulations, naive algorithm implementation (via numpy, pandas, statsmodels, scipy, matplotlib, seaborn, etc.), and sophisticated implementation through open-source library.

---
## Featured Projects & Competitions

* **Zillow's Home Value Prediction** [[Repository]](https://github.com/patrick-ytchou/KaggleZillowHomeValue)

| Model | Private Leaderboard Score | Private Leaderboard Ranking | Percentile (Top) |
| :---: | :---:| :---: | :---: |
| LightGBM | 0.07540 | 760 / 3770 | 20.2% |
| CatBoost | 0.07514 | 250 / 3770 | 6.6% |
| **Stacking** | **0.07505** | **120 / 3770** | **3.2%** |

* **Rossmann Store Sales** (Ongoing for Better Results)

| Model | Private Leaderboard Score | Private Leaderboard Ranking | Percentile (Top) |
| :---: | :---:| :---: | :---: |
| **LightGBM** | **0.11526** | **155 / 3303** | **4.7%** |


* **Yelp Review Analysis: Sentiment Analysis & Topic Modeling** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/1.%20Projects/YelpReviewAnalysis/Yelp%20Review%20Analysis%20--%20Sentiment%20Analysis%20%26%20Topic%20Modeling.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/1.%20Projects/YelpReviewAnalysis/Yelp%20Review%20Analysis%20--%20Sentiment%20Analysis%20%26%20Topic%20Modeling.ipynb)
    * Sentiment Analysis (Textblob, VADER, Afinn)| Topic Modeling | Natural Language Processing 

---
## Tableau Visualization
Data Visualization is a critical skill not only for data exploration but also for the decision making processes, and Tableau is one of the most widely used software for visualization. As such, starting from April 2020 I deicde to put aside one hour weekly to create hone my visualization skills and find unique insights from data.

[Workout Wednesday](http://www.workout-wednesday.com/) is a weekly data visualization challenge to help anyone interested in data visualization to build on the skills in Tableau. Each Wednesday a challenge is release and participants are asked to replicate the challenge that is posed as closely as possible. Thanks to this wonderful community, there are numerous resources that we can learn visualization from.

Check out my [Tableau Public Gallery](https://public.tableau.com/profile/yung.tang.chou#!/) if you are interesed in more informative visualization!

---
## Learning Notes

**Regression Analysis**
* **Regression Analysis: Assumptions for Linear Regression** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Assumptions%20for%20Linear%20Regression.ipynb)
    * Multicollinearity | Heteroscedasticity | Auto-Correlation
    * ResidualsPlot | White Test | Q-Q Plot | Durbin-Watson
    
* **Regression Analysis: Regression with Regularization** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Regression%20with%20Regularization.ipynb)
    * Naive Linear Regression | Regularization | Lasso/Ridege Regression
    
* **Regression Analysis: Parametric/Nonparametric regression** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Parametric%20and%20Nonparametric%20Regression.ipynb)
    * Polynomial Regression | Random Forest Regression | dtreeviz
    
* **Regression Analysis: Loss Functions for Regression Analysis** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Regression/Regression%20Analysis%20--%20Loss%20Functions%20for%20Regression%20Analysis.ipynb)
    * MSE | MAE | RMSE | MBE | MAPE | RMSLE | R² | Adjusted R²


**Classification & Clustering Analysis**

* **Decision Tree Classifier From Scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/DecisionTree/Decision%20Tree%20Classifier%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/DecisionTree/Decision%20Tree%20Classifier%20from%20Scratch.ipynb)
    * Gini Impurity | Parent/Child Nodes | Decision Tree | Tree Visualization


**Neural Network**
* **Tips for Neural Network Training** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/NeuralNetwork/Neural%20Network%20-%20Tips%20for%20Neural%20Network%20Training.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/NeuralNetwork/Neural%20Network%20-%20Tips%20for%20Neural%20Network%20Training.ipynb)
	* Activation Function | Optimizer | EarlyStopping | Regularization | Dropout

* **Convolutional Neural Network (CNN) with MNIST** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/NeuralNetwork/Neural%20Network%20-%20Convolutional%20Neural%20Network%20with%20MNIST.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/NeuralNetwork/Neural%20Network%20-%20Convolutional%20Neural%20Network%20%28CNN%29%20with%20MNIST.ipynb)
	* Convolution | Max Pooling | Flatten


**Optimization Techniques**

* **Genetic Algorithm from scratch: Traveling Salesman Problem** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Optimization/Genetic%20Algorithm%20from%20Scratch%20--%20Traveling%20Salesman%20Problem.ipynb)
    * Genetic Algorithm | Traveling Salesman Problem 

* **Gradient Descent from scratch** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Optimization/Gradient%20Descent%20from%20Scratch.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/2.%20Algorithms/Optimization/Gradient%20Descent%20from%20Scratch.ipynb)
	* Vanilla Gradient Descent | Adagrad | Stochastic Gradient Descent

**Model Explainability**

* **Model Explanation with Santander Dataset** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/3.%20Special%20Topics/ModelExplanation/Model%20Explanation%20with%20Santander%20Dataset.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/3.%20Special%20Topics/ModelExplanation/Model%20Explanation%20with%20Santander%20Dataset.ipynb)
	* Tree Visualization | Permutation Feature Importance | Partial Dependence Plot | SHAP Values

**Statistical Inference**

* **A/B Testing: Determine Sample Size** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/5.%20Statistical%20Inference/AB%20Testing/AB-Testing%20-%20Determine%20Sample%20Size.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/5.%20Statistical%20Inference/AB%20Testing/AB-Testing%20-%20Determine%20Sample%20Size.ipynb)
	* A/B Testing | Effect Size | Cohen's d

**Speical Topics**

* **Market Basket Analysis: Association Rule Mining** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/3.%20Special%20Topics/AssociationRules/Market%20Basket%20Analysis%20--%20Association%20Rule%20Explained.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/3.%20Special%20Topics/AssociationRules/Market%20Basket%20Analysis%20--%20Association%20Rule%20Explained.ipynb)
    * Market Basket Analysis | Apriori Algorithm


**Feature Engineering**

* **Imbalanced Data Manipulation** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/7.%20Feature%20Engineering/ImbalancedDataManipulation/Imbalanced%20Dataset%20Manipulation.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/7.%20Feature%20Engineering/ImbalancedDataManipulation/Imbalanced%20Dataset%20Manipulation.ipynb)
    * Imbalanced Data Manipulation | Tomek's Link | SMOTE | SMOTETomek


**Visualization Technique**

* **Zillow's Home Value Prediction Exploratory Data Analysis** [[Notebook]](https://github.com/patrick-ytchou/Data-Science/blob/master/6.%20Visualization/Zillow's%20Home%20Value%20Prediction%20Exploratory%20Data%20Analysis.ipynb) | [[nbviewer]](https://nbviewer.jupyter.org/github/patrick-ytchou/Data-Science/blob/master/6.%20Visualization/Zillow%27s%20Home%20Value%20Prediction%20Exploratory%20Data%20Analysis.ipynb)


---
## Pending Topics

* `Clustering Analysis`

* `Recommender System`

* `Support Vector Machines`

* `Entity Embedding`

---
## Data Science Resources 

Below is some wonderful data science resources collected online.


### Professional Skills:

[How To Speak With Confidence To Absolutely Anyone](https://medium.com/@georgejziogas/how-to-speak-with-confidence-to-absolutely-anyone-17aff65c37ef)




### Comprehensive Resource:

**Github:**

[firmai/data-science-career](https://github.com/firmai/data-science-career)

[conordewey3/DS-Career-Resources](https://github.com/conordewey3/DS-Career-Resources/)

[MaximAbramchuck/awesome-interview-questions](https://github.com/MaximAbramchuck/awesome-interview-questions)

[109 Data Science Interview Questions and Answers by Springboard](https://www.springboard.com/blog/data-science-interview-questions/)

[100+ Data Science and Machine Learning Interview Questions by LEARNDATASCI](https://www.learndatasci.com/data-science-interview-questions/)

**Medium:**

[The 4 fastest ways not to get hired as a data scientist](https://towardsdatascience.com/the-4-fastest-ways-not-to-get-hired-as-a-data-scientist-565b42bd011e)


### A/B Testing:

**Github:** 

[anantd/ab-testing](https://github.com/anantd/ab-testing)



### Data Science Tech Blogs:

[Airbnb Engineering & Data Science](https://medium.com/airbnb-engineering)

[AI³ | Theory, Practice, Business](https://medium.com/ai%C2%B3-theory-practice-business)

[Analytixon](https://analytixon.com/)

[Analytics Vidhya](https://www.analyticsvidhya.com/blog/)

[Applied Data Science](https://medium.com/applied-data-science)

[AWS Machine Learning Blog](https://aws.amazon.com/tw/blogs/machine-learning/)

[BCG Gamma Blog](https://medium.com/bcggamma)

[Berkeley Artificial Intelligence Blog](https://bair.berkeley.edu/blog/)

[Booking.com Data Science](https://booking.ai/)

[Cloudera Blog](https://blog.cloudera.com/)

[Cloudera FastForward Labs](https://blog.fastforwardlabs.com/)

[Cracking the Data Science Interview](https://medium.com/cracking-the-data-science-interview)

[Dair.ai](https://medium.com/dair-ai)

[Data Economy](https://dataconomy.com/)

[Data Driven Investor](https://medium.com/datadriveninvestor)

[Databrick Engineering Blog](https://databricks.com/blog/category/engineering)

[Dataiku Blog](https://blog.dataiku.com/)

[Data Series](https://medium.com/dataseries)

[Data Science @ Microosft](https://medium.com/data-science-at-microsoft)

[Data Science Lab @ Amsterdam](https://medium.com/data-science-lab-amsterdam)

[Data Science Weekly](https://www.datascienceweekly.org/)

[Data Science Student Community @ UC San Diego](https://medium.com/ds3ucsd)

[Domino Data Lab](https://blog.dominodatalab.com/)

[edureka! Blog](https://www.edureka.co/blog/)

[Facebook Research](https://research.fb.com/blog/)

[Flowing Data](https://flowingdata.com/)

[IBM Big Data & Analytics Hub](https://www.ibmbigdatahub.com/blogs)

[IBM Data and AI Medium Blog](https://medium.com/ibm-analytics)

[Indeed Engineering](https://medium.com/indeed-engineering)

[Insight Data Science](https://blog.insightdatascience.com/)

[Inside Machine Learning](https://medium.com/inside-machine-learning)

[Instacart Engineering](https://tech.instacart.com/)

[Instagram Engineering](https://instagram-engineering.com/)

[Kaggler Interviews & Highlights](https://medium.com/kaggle-blog)

[Lab 41](https://gab41.lab41.org/)

[Linkedin Engineering Blog](https://engineering.linkedin.com/blog)

[Machine Learning Mastery](https://machinelearningmastery.com/blog/)

[ManoMano Tech Blog](https://medium.com/manomano-tech)

[MLB Technology](https://technology.mlblogs.com/)

[Multiple Views Visualization Research Explained](https://medium.com/multiple-views-visualization-research-explained)

[Open Analytics](https://medium.com/opex-analytics)

[Oracle Data Science Blog](https://blogs.oracle.com/datascience/)

[Probably Overthinking it](https://www.allendowney.com/blog/)

[SAS Data Science Blog](https://blogs.sas.com/content/subconsciousmusings/)

[Sabastian Raschka](https://sebastianraschka.com/blog/index.html)

[Sabastian Ruder](https://ruder.io/)

[Sciforce Blog](https://medium.com/sciforce)

[Scribd Data Sciecne & Engirneering](https://medium.com/scribd-data-science-engineering)

[Sicara.ai](https://www.sicara.ai/blog/)

[Simon Frason University Professional Master's Program in Computer Science Student Publication](https://medium.com/
sfu-big-data)

[Simplystats](https://simplystatistics.org/)

[Square Corner](https://developer.squareup.com/blog/category/data-science/)

[StitchFix Blog](https://multithreaded.stitchfix.com/blog/)

[Tableau Blog](https://www.tableau.com/about/blog)

[The Unofficial Google Data Science Blog](http://www.unofficialgoogledatascience.com/)

[Toward Data Science](https://towardsdatascience.com/)

[Uber Enginneering](https://eng.uber.com/category/articles/ai/)

[Vantage AI](https://medium.com/vantageai)

[Wayfair Tech Blog](https://tech.wayfair.com/category/data-science/)

[Walmart Lab](https://medium.com/walmartlabs)

[Women in Big Data](https://www.womeninbigdata.org/blog/)

[Yelp Engineering Blog](https://engineeringblog.yelp.com/)