#!/usr/bin/env python
# coding: utf-8

# In[23]:


### FINAL PROJECT - PABLO GARCIA (13159411)

### 1. INITIAL STEPS

# First, I will import OS, scikit learn, keras and rest of libraries I will use during the feature selection and modelling
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'#I will use theano as keras backend
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.utils import np_utils


# In[19]:


# When running the code from a different machine, please change the file location
data_train = pd.read_csv (r'/Users/jose/Documents/MSc_Data_Science/Year_2/Final project/Data Scania trucks/aps_failure_training_set.csv', sep= ",")
data_test = pd.read_csv (r'/Users/jose/Documents/MSc_Data_Science/Year_2/Final project/Data Scania trucks/aps_failure_test_set.csv', sep= ",")
data_train.head()
print(data_train.dtypes)


# In[3]:


### 2. DATA PRE-PROCESSING

## Train dataset: I will first convert train object data into numeric data, except the target variable 'class'

list_col = list(data_train.iloc[:,1:171])
data_train[list_col] = data_train[list_col].apply(pd.to_numeric, errors="coerce")
print(data_train.dtypes)


# In[4]:


## MISSING VALUES
# Train dataset: Replacing the na values of the train dataset with the previous value using 'ffill'
data_train = data_train.fillna(method='ffill')
data_train.dropna(inplace = True)
data_train.isnull().sum().sum()
data_train.head()


# In[5]:


# Test dataset: I will now convert test object data into numeric data, except the target variable 'class'

list_col_t = list(data_test.iloc[:,1:171])
data_test[list_col_t] = data_test[list_col_t].apply(pd.to_numeric, errors="coerce")
print(data_test.dtypes)

# Replacing the na values from the test dataset with the previous value using 'ffill'

data_test = data_test.fillna(method='ffill')
data_test.dropna(inplace = True)
data_test.isnull().sum().sum()
data_test.head()


# In[6]:


## NORMALISATION
X_train_raw = pd.DataFrame(data_train.loc[:, list_col].values,columns = list_col)
y_train_raw = pd.DataFrame(data_train.loc[:,['class']].values,columns = ['class'])
X_test_raw = data_test.loc[:, list_col].values
y_test = data_test.loc[:, 'class'].values

mm_scaler = preprocessing.MinMaxScaler()
X_train_norm = pd.DataFrame(mm_scaler.fit_transform(X_train_raw),columns = list_col)
X_test = pd.DataFrame(mm_scaler.transform(X_test_raw), columns = list_col)


# In[7]:


## VISUALISATION
# SCATTER BOXPLOT
X_train_norm.boxplot(figsize=(40,40))
pyplot.show()


# In[8]:


# CORRELATION MATRIX
f = plt.figure(figsize=(40, 40))
plt.matshow(data_train.corr(), fignum=f.number)
plt.xticks(range(data_train.shape[1]), data_train.columns, fontsize=8, rotation=45)
plt.yticks(range(data_train.shape[1]), data_train.columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
plt.title('Correlation Matrix', fontsize=30);


# In[9]:


## REBALANCING - The dataset is not balanced, since we have 59000 negatives and 100 positives
target_count = data_train['class'].value_counts()
print('Class 0:', target_count['neg'])
print('Class 1:', target_count['pos'])
target_count.plot(kind='bar', title='Count (target)')


# In[10]:


# I have combined oversampling (SMOTE) with random undersampling.

# I will first oversample the minority class using the SMOTE method.
over_sampler = SMOTE(sampling_strategy='minority')

# Then I fit the object to our training data
X_smote, y_smote = over_sampler.fit_sample(X_train_norm,y_train_raw)

# Finally I undersample randomly the majority class
under_sampler = RandomUnderSampler(sampling_strategy='majority')

# The new X, Y and training dataset are consequently changed
X_train, y_train = under_sampler.fit_sample(X_smote, y_smote)
data_resampled = pd.concat([y_train, X_train], axis=1)
data_resampled.head()


# In[11]:


# I will count positive and negative class values to verify they are rebalanced
r_target_count = data_resampled['class'].value_counts()
print('Class 0:', r_target_count['neg'])
print('Class 1:', r_target_count['pos'])
r_target_count.plot(kind='bar', title='Count (target)')


# In[12]:


### 3. FEATURE SELECTION AND FEATURE EXTRACTION

## 3.1 FEATURE SELECTION

# 3.2.1 SelectKbest
# I will use the SelectKbest function with chi2 to do a statistical test
# and learn which are the best 10 features that provide more information. It works because I 
# do not have any negative values in the dataset
test = SelectKBest(score_func=chi2, k=15)
fit = test.fit(X_train, y_train)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X_train.columns)
# concatenate dataframes
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature_Name','Score']  # name output columns
print(feature_scores.nlargest(15,'Score'))  # print 15 best features


# In[13]:


## FEATURE EXTRACTION

# 3.2.1 PCA dimensionality reduction

pca = PCA(n_components=15)
fit_pca = pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)
# eigenvectors = pd.DataFrame(data = X_t_train
  #           , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
print("Explained Variance: %s" % fit_pca.explained_variance_ratio_)
print(fit_pca.components_)


# In[51]:


# 3.2.2 ExtraTrees classifier
model_xtrees = ExtraTreesClassifier()
model_xtrees.fit(X_train, y_train)

feature_importance = model_xtrees.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), feature_importance[indices],
        color="r", align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[14]:


### 4. ALGORITHM SELECTION

## 4.1. Classification with Logistic Regression
clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)
print("accuracy: {:.6f}".format(clf_log.score(X_test, y_test)))
y_pred = clf_log.predict(X_test)


# In[ ]:


## 4.2. Classification with KNN
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
print("accuracy: {:.6f}".format(clf_knn.score(X_test, y_test)))
y_pred = clf_knn.predict(X_test)


# In[ ]:


## 4.2. Classification with Decision Trees
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
print("accuracy: {:.6f}".format(clf_knn.score(X_test, y_test)))
y_pred = clf_knn.predict(X_test)


# In[22]:


# 4.3. Neural networks
# I define the keras model with the NN structure below
model = Sequential([
  Dense(64, activation='relu', input_shape=(170,)), 
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=30)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


# In[ ]:





# In[ ]:




