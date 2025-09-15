import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import set_config

df=pd.read_csv("tested.csv")

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

#Step-1-Train Test Split
x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=['Survived']),df['Survived'],
                                               test_size=0.2,random_state=42)

#Step-2-Missing values
# df.info()
p1=ColumnTransformer([
    ("fill_age",SimpleImputer(strategy="median"),[2]),
    ("fill_Fare",SimpleImputer(strategy="most_frequent"),[5])
],remainder="passthrough")

#Step-3-Covert categorical data
p2=ColumnTransformer([
    ("change_sex",OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])
])

#Step-4-Resize
p3=ColumnTransformer([
    ("scale",MinMaxScaler(),slice(0,10))
])

#Step-5-Train
p4=DecisionTreeClassifier(max_depth=5, random_state=42)

#Step-6-Pipeline
pipe=Pipeline([
    ("p1",p1),
    ("p2",p2),
    ("p3",p3),
    ("p4",p4)
])

set_config(display='diagram')
(pipe.fit(x_train,y_train))
y_pred=pipe.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred)*100%)



