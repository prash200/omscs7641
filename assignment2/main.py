#!/usr/bin/env python
# coding: utf-8

# # CS 7641 Assignment 2: Randomized Optimization

# In[10]:


import mlrose

from helpers import *
from opt_probs import *

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


ccp_problem = DiscreteOpt(length=64,
                          fitness_fn=mlrose.ContinuousPeaks(),
                          max_val=2)
ccp_learning_curves, ccp_best_fitnesses, ccp_execution_times = optimize(ccp_problem)


# In[251]:


plot_learning_curves(*ccp_learning_curves)


# In[252]:


plot_best_fitnesses(*ccp_best_fitnesses)


# In[253]:


plot_execution_times(*ccp_execution_times)


# In[67]:


edges = np.random.randint(64, size=(1024, 2))
kc_problem = DiscreteOpt(length=64,
                         fitness_fn=mlrose.MaxKColor(edges),
                         max_val=2,
                         maximize=False)
kc_learning_curves, kc_best_fitnesses, kc_execution_times = optimize(kc_problem)


# In[256]:


plot_learning_curves(*kc_learning_curves)


# In[255]:


plot_best_fitnesses(*kc_best_fitnesses)


# In[257]:


plot_execution_times(*kc_execution_times)


# In[89]:


coords = np.random.randint(100, size=(32, 2))
tsp_problem = TSPOpt(length=len(coords),
                     coords=coords,
                     maximize=False)
tsp_learning_curves, tsp_best_fitnesses, tsp_execution_times = optimize(tsp_problem)


# In[260]:


plot_learning_curves(*tsp_learning_curves)


# In[259]:


plot_best_fitnesses(*tsp_best_fitnesses)


# In[261]:


plot_execution_times(*tsp_execution_times)


# In[5]:


df = pd.read_csv("wind.csv", skipinitialspace=True)


# In[6]:


plot_hist(df, "binaryClass", {"binaryClass": "class"})


# In[7]:


X = StandardScaler().fit_transform(df.drop(['binaryClass'], axis='columns'))
y = LabelEncoder().fit_transform(df['binaryClass'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


nn_learning_curves, nn_best_fitnesses, nn_execution_times = fit(X_train, y_train)


# In[ ]:


plot_learning_curves(*nn_learning_curves, is_mimic=False)


# In[ ]:


plot_best_fitnesses(*nn_best_fitnesses, is_mimic=False)


# In[ ]:


plot_execution_times(*nn_execution_times, is_mimic=False)


# In[474]:


plot_evaluation_matrix(X_train, y_train, X_test, y_test,
                       restarts=0,
                       decay=0.92,
                       mutation_prob=0.2)

