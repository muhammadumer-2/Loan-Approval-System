import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Loan prediction.csv")
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())

sns.countplot(x="not.fully.paid", data=df)
plt.show()

numeric_df = df.select_dtypes(include=['float64', 'int64'])  
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

sns.countplot(x="purpose", data=df)
plt.xticks(rotation=90)
plt.show()

sns.boxplot(x="purpose", y="int.rate", data=df)
plt.xticks(rotation=90)
plt.show()

df["installment_to_income_ratio"] = (
    df["installment"] / df["log.annual.inc"]
)
df["credit_history"] = (df["delinq.2yrs"] + df["pub.rec"]) / df[
    "fico"]

df = df.drop(['credit.policy', 'days.with.cr.line', 'purpose'], axis=1)

le = LabelEncoder()

df['not.fully.paid'] = le.fit_transform(df['not.fully.paid'])

scaler = StandardScaler()

numerical_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec','credit_history','installment_to_income_ratio']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

sm = SMOTE(random_state=42)

X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

X_resampled, y_resampled = sm.fit_resample(X, y)

df = pd.concat([X_resampled, y_resampled], axis=1)
df['not.fully.paid'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print("Random Forest Classifier Accuracy: {:.2f}%".format(rf_score*100))

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.title("Loan Approval System")
st.header("")
st.write("Enter the details of loan applicant to check if the loan is approved or not.")
st.header("")
st.header("")
def predict_loan_status(interest_rate, installment, log_annual_income, dti_ratio, fico_score, revolving_balance, revolving_utilization, inquiries_last_6_months, delinquencies_last_2_years, public_records, installment_to_income_ratio, credit_history):
    # Sample prediction logic (replace with your actual prediction model)
    prediction = 1  # Example prediction result
    
    if prediction == 0:
        return "Loan fully paid"
    else:
        return "Loan not fully paid"

interest_rate = st.slider("Interest Rate", min_value=0.05, max_value=0.23, step=0.01)
installment = st.slider("Installment", min_value=100, max_value=950, step=10)
log_annual_income = st.slider("Log Annual Income", min_value=7.0, max_value=15.0, step=0.1)
dti_ratio = st.slider("DTI Ratio", min_value=0, max_value=40, step=1)
fico_score = st.slider("FICO Score", min_value=600, max_value=850, step=1)
revolving_balance = st.slider("Revolving Balance", min_value=0, max_value=120000, step=1000)
revolving_utilization = st.slider("Revolving Utilization", min_value=0, max_value=120, step=1)
inquiries_last_6_months = st.slider("Inquiries in Last 6 Months", min_value=0, max_value=10, step=1)
delinquencies_last_2_years = st.slider("Delinquencies in Last 2 Years", min_value=0, max_value=20, step=1)
public_records = st.slider("Public Records", min_value=0, max_value=10, step=2)
installment_to_income_ratio = st.slider("Installment to Income Ratio", min_value=0.0, max_value=5.0, step=0.1)
credit_history = st.slider("Credit History", min_value=0, max_value=10, step=1)

prediction_result = predict_loan_status(interest_rate, installment, log_annual_income, dti_ratio, fico_score, revolving_balance, revolving_utilization, inquiries_last_6_months, delinquencies_last_2_years, public_records, installment_to_income_ratio, credit_history)

if st.button("Predict Loan Status"):
    # Predict loan status based on selected values
    prediction_result = predict_loan_status(interest_rate, installment, log_annual_income, dti_ratio, fico_score, revolving_balance, revolving_utilization, inquiries_last_6_months, delinquencies_last_2_years, public_records, installment_to_income_ratio, credit_history)
    # Display prediction result
    st.write("Prediction Result:", prediction_result)













