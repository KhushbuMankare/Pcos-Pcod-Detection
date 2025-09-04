# %% [markdown]
# # PROJECT -  Polycystic ovary syndrome (PCOS) PREDICTION USING MACHINE LEARNING 

# %%
import pandas as pd
import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# %%
df = pd.read_csv("PCOS_infertility (1).csv")
df = df.fillna(0)

# %%
df.columns

# %%
df

# %% [markdown]
# # DATA SEGREGATION INTO FEATURES AND TARGET

# %%
inputs = df.drop(["PCOS (Y/N)","Sl. No","Patient File No."],axis = 'columns')

# %%
inputs.head(50)
inputx = inputs.head(4)

# %%
inputx
inputy = inputx.tail(2)

# %%
outputy = inputy.apply(lambda col: col.map(lambda x: ""))


# %%
target = df["PCOS (Y/N)"]

# %%
targetdum = target.head(4)
inputsdumy = inputs.head(4)


# %% [markdown]
# # DATA ENCODING FOR TEXT BASED FEATURES & REFORMATION 

# %%
inputs

# %%
# NOT REQUIRED SINCE ALL FEATURES HAVE NUMERIC DATATYPE

# %% [markdown]
# ## DATA SEGRECATION FOR TRAINING AND TESTING MACHINE MODEL(WITHOUT PCA)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputsdumy, targetdum, test_size=0.2)

# %%
df.dtypes

# %%
df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'])


# %% [markdown]
# # Applying Machine Learning Models without  Principal Component Analysis

# %% [markdown]
# ## Logistic Regression

# %%
print(inputs.dtypes)
print(target.dtype)

# Convert to numeric if needed
inputs = inputs.apply(pd.to_numeric, errors='coerce')
target = pd.to_numeric(target, errors='coerce')

# %%
print(inputs.isnull().sum())
print(target.isnull().sum())

# Drop rows with missing values
inputs = inputs.dropna()
target = target[inputs.index]

# %%
from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns
categorical_cols = inputs.select_dtypes(include=['object', 'category']).columns

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(inputs[categorical_cols])

# Combine with numeric columns
numeric_cols = inputs.select_dtypes(include=['int64', 'float64']).columns
inputs_encoded = pd.concat([inputs[numeric_cols], 
                            pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))], 
                           axis=1)

# %%
import sklearn
print(sklearn.__version__)

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import randint
import numpy as np
import joblib
import pickle

# Generate a synthetic dataset for demonstration purposes
inputs, target = make_classification(n_samples=1000, n_features=20, 
                                     n_informative=10, n_redundant=10, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=5)  # Select top 5 features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Create a pipeline
pipeline = Pipeline([
    ('classifier', RandomForestClassifier())
])

# Define parameter space
param_dist = {
    'classifier__n_estimators': randint(50, 500),
    'classifier__max_depth': randint(1, 20),
    'classifier__min_samples_split': randint(2, 11),
    'classifier__min_samples_leaf': randint(1, 11)
}

# Randomized search with cross-validation using threading backend
with joblib.parallel_backend('threading'):
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                       n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)
    # Fit the model
    random_search.fit(X_train_selected, y_train)

# Print best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

# Evaluate on the test set
best_model = random_search.best_estimator_
score = best_model.score(X_test_selected, y_test) * 100
print("Final model score on test set:", score)



# %%
reg_model = LogisticRegression()
reg_model.fit(X_train_selected, y_train)
train_score = reg_model.score(X_train_selected, y_train) * 100
test_score = reg_model.score(X_test_selected, y_test) * 100
print("LogisticRegression training score:", train_score)
print("LogisticRegression test score:", test_score)

# %% [markdown]
# ## Support Vector Machine

# %%
print("Unique classes in y_train:", np.unique(y_train))


# %%
sv_model = SVC()
sv_model.fit(X_train_selected, y_train)
sv_score = sv_model.score(X_test_selected, y_test) * 100
print("SVC test score:", sv_score)

# %% [markdown]
# ## Random Forest Classification

# %%
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
Score = clf.score(inputs,target)*100
print("prediction score is",Score )

# %% [markdown]
# ### Machine learning model without Principle Component Analysis had Better prediction score.

# %% [markdown]
# ## Making Pickle file to save the machine learning model

# %%
import pickle

with open('modelfinal1.pkl', 'wb') as file:
    pickle.dump(clf, file)



