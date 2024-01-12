##########  EMAIL SPAM DETECTION  ##########

print('EMAIL SPAM DETECTION')

#importing necessory libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('task4spamemaildata.csv' , encoding = 'ISO-8859-1') #reading dataset
# print dataset
df.head()
df.tail()
print(df)
df.shape
df.size 
df.info()  # give inforamtion about dataset
df.describe()   #give Description about dataset
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df
df=df.rename(columns={'v1':'Target','v2':'Message'})
df.isnull().sum()
df.drop_duplicates(keep='first',inplace=True)
df.duplicated().sum()
df.size
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
df['Target']
df.head()
plt.pie(df['Target'].value_counts(), labels = ['ham', 'spam'], autopct = "%0.2f")
plt.show()
x=df['Message']
y=df['Target']
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)  #splitting data into train and test
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn import svm
cv=CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)
print(x_train_cv)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression() #creating model
lr.fit(x_train_cv,y_train)
prediction_train=lr.predict(x_train_cv)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, prediction_train)*100)
prediction_test = lr.predict(x_test_cv)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction_test)*100)