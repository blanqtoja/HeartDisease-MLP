import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
y.columns = ['target']

data = pd.concat([X, y], axis=1 )

# print(data)

def boxplot_analysis(data):
    numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

    eda_df = data.loc[:, numeric_features].copy()

    plt.figure(figsize=(16, 10))

    for i in range(len(eda_df.columns)):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(eda_df[eda_df.columns[i]])

    plt.show()

def heatmap_analysis(data):
    numeric_features = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']

    eda_df = data.loc[:, numeric_features].copy()

    corr = eda_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
    plt.title("Correlation Heatmap")
    plt.show()

def pie_chart_analysis(data):
    # Pie chart for class distribution
    # 0 = No Heart Disease
    # 1, 2, 3, 4 = Heart Disease
    plt.figure(figsize=(8, 8))
    # print(data['target'].value_counts())
    plt.pie(data['target'].value_counts(), labels=[0, 1, 2, 3, 4], autopct='%.1f%%', colors=['#44ce1b', '#bbdb44', '#f7e379', '#f2a134', '#e51f1f'])
    plt.title("Class Distribution")
    plt.show()

# boxplot_analysis(data)
# heatmap_analysis(data)
# pie_chart_analysis(data)

#zamiana wartosci 1,2,3,4 na 1, zmiana problemu na binarny
data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)


#preprocessing
def onehot_encode(df, column_dict):
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    # print(df)
    return df

def preprocess_inputs(df, scaler):
    df = df.copy()
    
    # One-hot encode the nominal features
    nominal_features = ['cp', 'slope', 'thal']
    df = onehot_encode(df, dict(zip(nominal_features, ['CP', 'SL', 'TH'])))
    
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    
    # Scale X
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y


# X, y = preprocess_inputs(data, MinMaxScaler())
# X, y = preprocess_inputs(data, StandardScaler())
X, y = preprocess_inputs(data, RobustScaler())


#training


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
def train_models(X_train, y_train):
        
    lm = LogisticRegression()
    lm.fit(X_train, y_train)

    svm = SVC()
    svm.fit(X_train, y_train)

    mlp = MLPClassifier(max_iter=700) # mlp nie zbiega sie w 200, ani w 500 iteracjach, zbiega sie przy okolo 1000, ale najlepsze wyniki dostalismy dla iter = 800
    mlp.fit(X_train, y_train)

    return lm, svm, mlp

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

X_test = X_test.dropna()
y_test = y_test[X_test.index]

X_train = X_train.values
X_test = X_test.values

# wypelnienei wartosci NaN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# przy danych skalowanych minmax

lm, svm, mlp = train_models(X_train, y_train)

#accuracy
print("Logistic Regression Accuracy: {:.2f}%".format(lm.score(X_test, y_test) * 100))
print("Support Vector Machine Accuracy: {:.2f}%".format(svm.score(X_test, y_test) * 100))
print("Neural Network Accuracy: {:.2f}%".format(mlp.score(X_test, y_test) * 100))