import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv")


outcome_labels = train['Outcome'].value_counts().index
outcome_values = train['Outcome'].value_counts().values

plt.figure(figsize= (6,4))
sns.barplot(x= outcome_labels, y=outcome_values)

plt.figure(figsize=(6,4))
glucose_hist = train["Glucose"]
sns.histplot(x=glucose_hist)
plt.title("Distribution of Glucose Levels")

plt.figure(figsize=(3,4))
BMI_box = train["BMI"]
sns.boxplot(y=BMI_box)
plt.ylabel("BMI")
plt.title("Outlier Detection of BMI")

plt.figure(figsize=(5,4))
x_Glucose_scatter = train["Glucose"]
y_Insulin_scatter = train["Insulin"]
sns.scatterplot(x = x_Glucose_scatter, y= y_Insulin_scatter)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.title("Relationship Between Glucose and Insulin")

plt.figure(figsize=(6,4))
BloodPressure_hist = train['BloodPressure']
sns.histplot(x=BloodPressure_hist)
plt.title("Distribution of Blood Pressure Values")

plt.figure(figsize=(3,4))
BloodPressure_box = train['BloodPressure']
sns.boxplot(y=BloodPressure_box)
plt.ylabel('BloodPressure')
plt.title('Outlier Detection of BloodPressure')
plt.show()


def categorize_blood_pressure(bp):
    if bp <= 60:
        return "Low"
    elif bp >= 80:
        return "High"
    else:
        return "Normal"

train["BP_Category_apply"] = train["BloodPressure"].apply(categorize_blood_pressure)
test["BP_Category_apply"] = test["BloodPressure"].apply(categorize_blood_pressure)

bp_mapping = {"Low": 0, "Normal": 1, "High": 2}

train["BP_Category_apply"] = train["BP_Category_apply"].map(bp_mapping).astype("int64")
test["BP_Category_apply"] = test["BP_Category_apply"].map(bp_mapping).astype("int64")
train = train.drop(columns=["BloodPressure"])
test = test.drop(columns=["BloodPressure"])

X_train = train.drop(columns=['Outcome','ID'])
y_train = train['Outcome']
X_test = test.drop(columns=["ID"])

Glucose_median = X_train["Glucose"].median()
SkinThickness_median = X_train["SkinThickness"].median()
Insulin_median = X_train["Insulin"].median()
BMI_median = X_train["BMI"].median()

X_train["Glucose"] = X_train["Glucose"].replace(0,Glucose_median)
X_train["SkinThickness"] = X_train["SkinThickness"].replace(0,SkinThickness_median)
X_train["Insulin"] = X_train["Insulin"].replace(0,Insulin_median)
X_train["BMI"] = X_train["BMI"].replace(0,BMI_median)

X_test["Glucose"] = X_test["Glucose"].replace(0,Glucose_median)
X_test["SkinThickness"] = X_test["SkinThickness"].replace(0,SkinThickness_median)
X_test["Insulin"] = X_test["Insulin"].replace(0,Insulin_median)
X_test["BMI"] = X_test["BMI"].replace(0,BMI_median)


numeric_columns = ["Pregnancies", "Glucose","SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction","Age"]
scaler = StandardScaler()

X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

model = DecisionTreeClassifier(max_depth =3)
model.fit(X_train, y_train)
y_test = model.predict(X_test)

sample["Outcome"]= y_test
sample.to_csv("sample.csv", index=False)
