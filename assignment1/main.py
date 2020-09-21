#!/usr/bin/env python
# coding: utf-8

# # CS 7641 Assignment 1: Supervised Learning

# This Notebook contains answers to CS7641 Assignment 1: Supervised Learning.
# 
# 
# It compares and contrasts 5 Supervised Machine Learning classfication algorithms:
# - **Decision trees**
# - **Boosting**
# - **Support vector machines (2 kernels: poly and rbf)**
# - **Neural networks**
# - **K nearest neighbors**
# 
# On 2 Datasets:
# - **Wind speed:** https://www.openml.org/d/503 and https://www.openml.org/d/847
# - **Customer retaintion:** https://www.openml.org/d/42178
# 
# For each algorithm there are following outputs:
# - **Model complexity curves:** A model with given number of training instances is trained with different values of hyperparameters and training and cross-validation (CV) scores are plotted against different values of hyperparameters such that complexity of model increases left to right. Typically, as model starts becoming more complex both training score and CV scores improve. But later, as model becomes overly complex CV score decreases while training score keeps on improving, which is an indicator of overfitting. Model complexity curves helps us fine tune the optimal values of hyperparameters such the trained model is complex enough but not too complex.
# - **Learning curves:** A model with given values of hyperparameters is trained with increasing number of trainig instances and resulting training and cross-validation (CV) scores and times are plotted as function of number of training instances. Typically, these curves rises sharply initially and then flattens out. These curves helps us understand if and how number of training instances are helping in improving scores and/or times. Which inturn helps us decide if geting new instances for training will improve the model or the hyperparameters needs to be retuned.
# - **Out-of-sample model evaluation metrics:** The CV scores are not true estimates of the model performance on unseen data. As CV set itself was used to fine tune hyperpamater values, CV scores are typically overestimates of performance on unseen data. To get a true estimate of model performance on unseen data, a small set of training data is held-out at the beginning and the model performance is reported on this set.
# 
# Lastly, outputs from each of the above algorithms are compared based on:
# - **Cross validation scores:** CV score for each of the tuned and trained model is plotted as a function of number of training instances. Typically, as number of training instances increases CV scores increases as well. These curves helps us compare the rate of increase of CV scores with number of training instances.
# - **Training times:** Training times for each of the tuned and trained model is plotted as a function of number of training instances. Typically, training times also grows with number of training instances. These curves helps us compare the rate of increase of training times with number of training instances. These 2 types of curves together help us understand and strike a balance between model performance and time required.
# - **Model evaluation metrics:** Model evaluation metrics for each tuned and trained model is compared. This gives a true estimate of model performance on unseen data.

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.display.max_columns = None

from helpers import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Notes on implementation:
# - To keep this notebook ligible, most of the code is abstracted out in helpers.py and imported in this notebook.
# - Following 3 steps are performed on each dataset mentioned above:
#     - **Preprocessing:**
#         - Data is loaded from csv files downloaded from sources mentioned in README.md.
#         - Null values (if any) are handled.
#         - LabelEncoding is performed on target variable.
#         - OneHotEncoding is performed on categorically variables (if any).
#         - StandardScaling is performed on all variables.
#         - Data is split in X and y.
#         - Data is split in train and test sets.
#     - **Machine learning:**
#         - 6 models mentioned above are trained on each dataset mentioned above.
#         - RandomSearchCV to used to fit hyperparameters.
#         - 10-fold cross-valdation (CV) is performed during RandomSearchCV.
#         - Model complexity curves are plotted.
#         - Learning curves are plotted for the "best" model found.
#         - Finally, model evaluation metrics are calculated and printed for out-of-sample held-out set.
#     - **Model comparision:**
#         - After all 6 machine learning model are successfully trained and tuned; CV scores, training times and evaluation metrics are plotted.

# ## 1. Wind speed dataset

# ### 1.a. Preprocessing

# In[2]:


df = pd.read_csv("wind.csv", skipinitialspace=True)


# In[3]:


df.isnull().sum()


# In[4]:


df.head()


# In[5]:


X = df.drop(['binaryClass'], axis='columns')
y = pd.DataFrame(LabelEncoder().fit_transform(df['binaryClass']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### 1.b. Machine learning

# #### 1.b.i. Decision Tree

# In[6]:


decision_tree = fit_model([], DecisionTreeClassifier(random_state=42),
                          {'min_samples_split': range(2, 2000, 20)}, X_train, y_train[0])


# In[7]:


decision_tree_complexity = get_model_complexity(decision_tree, "dsec", "Decision Tree")
plot_model_complexity(decision_tree_complexity, "o-")


# #### <center>Fig 1. Model Complexity Curves: Decision Tree</center>

# In[8]:


decision_tree_learning_curve = get_learning_curve(decision_tree, X_train, y_train, "Decision Tree")
plot_learning_curve(decision_tree_learning_curve, "o-")


# #### <center>Fig 2. Learning Curves: Decision Tree</center>

# In[9]:


decision_tree_evaluation_metrics = get_model_evaluation_metrics(decision_tree, X_test, y_test, "Decision Tree")
print_model_evaluation_metrics(decision_tree_evaluation_metrics)


# #### <center>Table 1. Evaluation Metrics: Decision Tree</center>

# #### 1.b.ii. Boosting

# In[10]:


boosting = fit_model([], GradientBoostingClassifier(n_estimators=2, random_state=42),
                          {'min_samples_split': range(2, 2000, 20)}, X_train, y_train[0])


# In[11]:


boosting_complexity = get_model_complexity(boosting, "dsec", "Boosting")
plot_model_complexity(boosting_complexity, "s-")


# #### <center>Fig 3. Model Complexity Curves: Boosting</center>

# In[12]:


boosting_learning_curve = get_learning_curve(boosting, X_train, y_train, "Boosting")
plot_learning_curve(boosting_learning_curve, "s-")


# #### <center>Fig 4. Learning Curves: Boosting</center>

# In[13]:


boosting_evaluation_metrics = get_model_evaluation_metrics(boosting, X_test, y_test, "Boosting")
print_model_evaluation_metrics(boosting_evaluation_metrics)


# #### <center>Table 2. Evaluation Metrics: Boosting</center>

# #### 1.b.iii. SVM (Kernel: poly)

# In[14]:


polysvc = fit_model([], SVC(kernel='poly', degree=5, random_state=42),
                {'C': np.linspace(0.0005, 40.0, 500)}, X_train, y_train[0])


# In[15]:


polysvc_complexity = get_model_complexity(polysvc, "asc", "SVM (Kernel: poly)")
plot_model_complexity(polysvc_complexity, "D-")


# #### <center>Fig 5. Model Complexity Curves: SVM (Kernel: poly)</center>

# In[16]:


polysvc_learning_curve = get_learning_curve(polysvc, X_train, y_train, "SVM (Kernel: poly)")
plot_learning_curve(polysvc_learning_curve, "D-")


# #### <center>Fig 6. Learning Curves: SVM (Kernel: poly)</center>

# In[17]:


polysvc_evaluation_metrics = get_model_evaluation_metrics(polysvc, X_test, y_test, "SVM (Kernel: poly)")
print_model_evaluation_metrics(polysvc_evaluation_metrics)


# #### <center>Table 3. Evaluation Metrics: SVM (Kernel: poly)</center>

# #### 1.b.iv. SVM (Kernel: rbf)

# In[18]:


rbfsvc = fit_model([], SVC(kernel='rbf', random_state=42),
                {'C': np.linspace(0.0005, 10.0, 500)}, X_train, y_train[0])


# In[19]:


rbfsvc_complexity = get_model_complexity(rbfsvc, "asc", "SVM (Kernel: rbf)")
plot_model_complexity(rbfsvc_complexity, "^-")


# #### <center>Fig 7. Model Complexity Curves: SVM (Kernel: rbf)</center>

# In[20]:


rbfsvc_learning_curve = get_learning_curve(rbfsvc, X_train, y_train, "SVM (Kernel: rbf)")
plot_learning_curve(polysvc_learning_curve, "^-")


# #### <center>Fig 8. Learning Curves: SVM (Kernel: rbf)</center>

# In[21]:


rbfsvc_evaluation_metrics = get_model_evaluation_metrics(rbfsvc, X_test, y_test, "SVM (Kernel: rbf)")
print_model_evaluation_metrics(rbfsvc_evaluation_metrics)


# #### <center>Table 4. Evaluation Metrics: SVM (Kernel: rbf)</center>

# #### 1.b.v. Neural Network

# In[ ]:


mlp = fit_model([], MLPClassifier(activation='logistic', learning_rate_init=0.005, max_iter=500, random_state=42),
                {"hidden_layer_sizes": [(s,) for s in range(1, 21, 1)]}, X_train, y_train[0])


# In[ ]:


mlp_complexity = get_model_complexity(mlp, "asc", "Neural Network")
plot_model_complexity(mlp_complexity, "P-")


# #### <center>Fig 9. Model Complexity Curves: Neural Network</center>

# In[ ]:


mlp_learning_curve = get_learning_curve(mlp, X_train, y_train, "Neural Network")
plot_learning_curve(mlp_learning_curve, "P-")


# #### <center>Fig 10. Learning Curves: Neural Network</center>

# In[ ]:


mlp_evaluation_metrics = get_model_evaluation_metrics(mlp, X_test, y_test, "Neural Network")
print_model_evaluation_metrics(mlp_evaluation_metrics)


# #### <center>Table 5. Evaluation Metrics: Neural Network</center>

# #### 1.b.vi. K Nearest Neighbors

# In[ ]:


knn = fit_model([], KNeighborsClassifier(n_jobs=-1), {'n_neighbors': range(1, 100)}, X_train, y_train[0])


# In[ ]:


knn_complexity = get_model_complexity(knn, "dsec", "K Nearest Neighbors")
plot_model_complexity(knn_complexity, "v-")


# #### <center>Fig 11. Model Complexity Curves: K Nearest Neighbors</center>

# In[ ]:


knn_learning_curve = get_learning_curve(knn, X_train, y_train, "K Nearest Neighbors")
plot_learning_curve(knn_learning_curve, "v-")


# #### <center>Fig 12. Learning Curves: K Nearest Neighbors</center>

# #### Note: Cross-validation time is more than training time for KNN, which is atypical of any other algorithm we studied here.

# In[ ]:


knn_evaluation_metrics = get_model_evaluation_metrics(knn, X_test, y_test, "K Nearest Neighbors")
print_model_evaluation_metrics(knn_evaluation_metrics)


# #### <center>Table 6. Evaluation Metrics: K Nearest Neighbors</center>

# ### 1.c. Model Comparisions

# In[ ]:


plot_learning_curve_comparision([decision_tree_learning_curve,
                                 boosting_learning_curve,
                                 polysvc_learning_curve,
                                 rbfsvc_learning_curve,
                                 mlp_learning_curve,
                                 knn_learning_curve])


# #### <center>Fig 13. Model Comparisions</center>

# In[ ]:


print_model_evaluation_metrics_comparision([decision_tree_evaluation_metrics,
                                            boosting_evaluation_metrics,
                                            polysvc_evaluation_metrics,
                                            rbfsvc_evaluation_metrics,
                                            mlp_evaluation_metrics,
                                            knn_evaluation_metrics])


# #### <center>Table 7. Evaluation Metrics Comparisions</center>

# ## 2. Customer retaintion dataset

# ### 2.a. Data load and preprocessing

# In[ ]:


df = pd.read_csv("customer.csv", skipinitialspace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df = df.dropna()
df.head()


# #### Note: Since there are only 11 rows that has null values, they are dropped to handle missing values.

# In[ ]:


X = df.drop(['customerID', "Churn"], axis='columns')
y = pd.DataFrame(LabelEncoder().fit_transform(df['Churn']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### 2.b. Machine learning

# #### 2.b.i. Decision Tree

# In[ ]:


decision_tree = fit_model(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], DecisionTreeClassifier(random_state=42),
                          {'min_samples_split': range(2, 2000, 20)}, X_train, y_train[0])


# In[ ]:


decision_tree_complexity = get_model_complexity(decision_tree, "dsec", "Decision Tree")
plot_model_complexity(decision_tree_complexity, "o-")


# #### <center>Fig 14. Model Complexity Curves: Decision Tree</center>

# In[ ]:


decision_tree_learning_curve = get_learning_curve(decision_tree, X_train, y_train, "Decision Tree")
plot_learning_curve(decision_tree_learning_curve, "o-")


# #### <center>Fig 15. Learning Curves: Decision Tree</center>

# In[ ]:


decision_tree_evaluation_metrics = get_model_evaluation_metrics(decision_tree, X_test, y_test, "Decision Tree")
print_model_evaluation_metrics(decision_tree_evaluation_metrics)


# #### <center>Table 8. Evaluation Metrics: Decision Tree</center>

# #### 2.b.ii. Boosting

# In[ ]:


boosting = fit_model(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], GradientBoostingClassifier(n_estimators=40, random_state=42),
                          {'min_samples_split': range(2, 200, 2)}, X_train, y_train[0])


# In[ ]:


boosting_complexity = get_model_complexity(boosting, "dsec", "Boosting")
plot_model_complexity(boosting_complexity, "s-")


# #### <center>Fig 16. Model Complexity Curves: Boosting</center>

# In[ ]:


boosting_learning_curve = get_learning_curve(boosting, X_train, y_train, "Boosting")
plot_learning_curve(boosting_learning_curve, "s-")


# #### <center>Fig 17. Learning Curves: Boosting</center>

# In[ ]:


boosting_evaluation_metrics = get_model_evaluation_metrics(boosting, X_test, y_test, "Boosting")
print_model_evaluation_metrics(boosting_evaluation_metrics)


# #### <center>Table 9. Evaluation Metrics: Boosting</center>

# #### 2.b.iii. SVM (Kernel: poly)

# In[ ]:


polysvc = fit_model(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], SVC(kernel='poly', degree=5, random_state=42),
                {'C': np.linspace(0.0005, 5.0, 500)}, X_train, y_train[0])


# In[ ]:


polysvc_complexity = get_model_complexity(polysvc, "asc", "SVM (Kernel: poly)")
plot_model_complexity(polysvc_complexity, "D-")


# #### <center>Fig 18. Model Complexity Curves: SVM (Kernel: poly)</center>

# In[ ]:


polysvc_learning_curve = get_learning_curve(polysvc, X_train, y_train, "SVM (Kernel: poly)")
plot_learning_curve(polysvc_learning_curve, "D-")


# #### <center>Fig 19. Learning Curves: SVM (Kernel: poly)</center>

# In[ ]:


polysvc_evaluation_metrics = get_model_evaluation_metrics(polysvc, X_test, y_test, "SVM (Kernel: poly)")
print_model_evaluation_metrics(polysvc_evaluation_metrics)


# #### <center>Table 10. Evaluation Metrics: SVM (Kernel: poly)</center>

# #### 2.b.iv. SVM (Kernel: rbf)

# In[ ]:


rbfsvc = fit_model(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], SVC(kernel='rbf', random_state=42),
                {'C': np.linspace(0.0005, 5.0, 500)}, X_train, y_train[0])


# In[ ]:


rbfsvc_complexity = get_model_complexity(rbfsvc, "asc", "SVM (Kernel: rbf)")
plot_model_complexity(rbfsvc_complexity, "^-")


# #### <center>Fig 20. Model Complexity Curves: SVM (Kernel: rbf)</center>

# In[ ]:


rbfsvc_learning_curve = get_learning_curve(rbfsvc, X_train, y_train, "SVM (Kernel: rbf)")
plot_learning_curve(polysvc_learning_curve, "^-")


# #### <center>Fig 21. Learning Curves: SVM (Kernel: rbf)</center>

# In[ ]:


rbfsvc_evaluation_metrics = get_model_evaluation_metrics(rbfsvc, X_test, y_test, "SVM (Kernel: rbf)")
print_model_evaluation_metrics(rbfsvc_evaluation_metrics)


# #### <center>Table 11. Evaluation Metrics: SVM (Kernel: rbf)</center>

# #### 2.b.v. Neural Network

# In[ ]:


mlp = fit_model(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], MLPClassifier(activation='logistic', learning_rate_init=0.002, max_iter=500, random_state=42),
                {"hidden_layer_sizes": [(s,) for s in range(1, 21, 1)]}, X_train, y_train[0])


# In[ ]:


mlp_complexity = get_model_complexity(mlp, "asc", "Neural Network")
plot_model_complexity(mlp_complexity, "P-")


# #### <center>Fig 22. Model Complexity Curves: Neural Network</center>

# In[ ]:


mlp_learning_curve = get_learning_curve(mlp, X_train, y_train, "Neural Network")
plot_learning_curve(mlp_learning_curve, "P-")


# #### <center>Fig 23. Learning Curves: Neural Network</center>

# In[ ]:


mlp_evaluation_metrics = get_model_evaluation_metrics(mlp, X_test, y_test, "Neural Network")
print_model_evaluation_metrics(mlp_evaluation_metrics)


# #### <center>Table 12. Evaluation Metrics: Neural Network</center>

# #### 2.b.vi. K Nearest Neighbors

# In[ ]:


knn = fit_model(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], KNeighborsClassifier(n_jobs=-1),
                {'n_neighbors': range(1, 140)}, X_train, y_train[0])


# In[ ]:


knn_complexity = get_model_complexity(knn, "dsec", "K Nearest Neighbors")
plot_model_complexity(knn_complexity, "v-")


# #### <center>Fig 24. Model Complexity Curves: K Nearest Neighbors</center>

# In[ ]:


knn_learning_curve = get_learning_curve(knn, X_train, y_train, "K Nearest Neighbors")
plot_learning_curve(knn_learning_curve, "v-")


# #### <center>Fig 25. Learning Curves: K Nearest Neighbors</center>

# #### Note: Cross-validation time is more than training time for KNN, which is atypical of any other algorithm we studied here.

# In[ ]:


knn_evaluation_metrics = get_model_evaluation_metrics(knn, X_test, y_test, "K Nearest Neighbors")
print_model_evaluation_metrics(knn_evaluation_metrics)


# #### <center>Table 13. Evaluation Metrics: K Nearest Neighbors</center>

# ### 2.c. Model Comparisions

# In[ ]:


plot_learning_curve_comparision([decision_tree_learning_curve,
                                 boosting_learning_curve,
                                 polysvc_learning_curve,
                                 rbfsvc_learning_curve,
                                 mlp_learning_curve,
                                 knn_learning_curve])


# #### <center>Fig 26. Model Comparisions</center>

# In[ ]:


print_model_evaluation_metrics_comparision([decision_tree_evaluation_metrics,
                                            boosting_evaluation_metrics,
                                            polysvc_evaluation_metrics,
                                            rbfsvc_evaluation_metrics,
                                            mlp_evaluation_metrics,
                                            knn_evaluation_metrics])


# #### <center>Table 14. Evaluation Metrics Comparisions</center>

# ## 3. Conclusions

# #### Following comclusions can be drawn after analysing 6 different model on 2 data sets, mentioned above:
# - As model complexity increases, both training scores and CV scores improves. But later, as model becomes too complex CV score decreases while training score keeps on improving. This is point where the model starts overfitting **(Figs. 1, 3, 5, 7, 9, 11, 14, 16, 18, 20, 22 and 24)**. This is true for all algorithms across both datasets.
# - Initially, CV score learning curves sharply rise indicating improvements resulted from new training instances. After a few 1000s of training instances a point is reached after which negligible improvments in CV scores is observed **(Figs. 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23 and 25)**. This is true for all algorithms across both datasets.
# - Training time learning curves also rise with number of training instances. **(Figs. 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23 and 25)**. But the rate of increase is different for different algorithms. SVM (both poly and rbf kernels) and Nerural Network seems to be polynomial whereas other look like linear in number of training instances **(Figs. 13 and 26)**.
# - As more number of training instances are added, the algorithm used becomes irrelavant. This is evident from the fact that the CV scores and model performance on unseen data is almost equal for all the algorithms across both the datasets **(Figs. 13 and 26) and (Tables. 7 and 14)**.
# - 10-fold CV leads to good generalization over unseen data, as model performance over unseen data for each model above is almost same as CV score for that model **(Tables. 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12 and 13)**.
# - Performing feature selection can help in imporving CV scores decreasing training times.
# - Better hyperparameters can be found by doing more exhaustive grid search over multiple hyperparameters. But doing so would be time consuming.
