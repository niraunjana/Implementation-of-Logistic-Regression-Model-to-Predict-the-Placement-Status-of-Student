# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NIRAUNJANA GAYATHRI G R
RegisterNumber:  22008369
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print("Placement data")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
print("print data")
data1

x = data1.iloc[:,:-1]
print("Data-status")
x

y = data1["status"]
print("data-status")
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(" y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
![image](https://user-images.githubusercontent.com/119395610/235344682-24108ebc-7c4b-4a6b-adc3-aeba1a0734b4.png)

![image](https://user-images.githubusercontent.com/119395610/235344690-30edfdc4-3ddf-478d-9409-effc60acac99.png)

![image](https://user-images.githubusercontent.com/119395610/235344707-28c39344-cd66-4b09-a4e5-ddce01a98ecd.png)

![image](https://user-images.githubusercontent.com/119395610/235344720-ef608bb3-b5ab-4acd-b2e1-97228322fe79.png)

![image](https://user-images.githubusercontent.com/119395610/235344730-f4676f1c-b642-40ec-be3b-17cf27bfc149.png)

![image](https://user-images.githubusercontent.com/119395610/235344743-028d7ded-635b-4e85-aa6e-2f8cd91751de.png)

![image](https://user-images.githubusercontent.com/119395610/235344752-da829ed5-f09f-4dda-a183-f1e109674716.png)

![image](https://user-images.githubusercontent.com/119395610/235344759-6a8581a7-70b7-4bb7-9181-b8adce137ec4.png)

![image](https://user-images.githubusercontent.com/119395610/235344772-72ea722e-0020-489f-be61-ecbcac55e2e6.png)

![image](https://user-images.githubusercontent.com/119395610/235344778-5121373f-147e-40ee-aca0-f21017c0c649.png)

![image](https://user-images.githubusercontent.com/119395610/235344787-3d80576f-74a5-4ce8-a418-3798a477ced9.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
