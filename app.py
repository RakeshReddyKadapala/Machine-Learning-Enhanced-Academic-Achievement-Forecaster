import streamlit as st
from db import db_connect

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def run_models(background_path, result_path, employment_path, target):
    # Assuming all datasets can be read and merged based on a common key; adjust as necessary.
    bg_df = pd.read_excel('background.xlsx')
    result_df = pd.read_excel('result.xlsx')
    employment_df = pd.read_excel('employement.xls')
    
    # Merging dataframes; adjust this based on how you want to combine your datasets.
    # This is a placeholder step, and you may need to merge on specific columns.
    df = pd.concat([bg_df, result_df, employment_df], axis=1)
    
    # Splitting dataset into features (X) and target (y)
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Defining models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier()
    }
    
    # Training and evaluating models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
    
    return results




# to start type streamlit run app.py

st.title('Student Performance Analysis')
global metrics
metrics = [None,'Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5']
dataset_names = [None,'background', 'employeement', 'result']

# A dropdown menu for selecting the dataset not in the sidebar
dataset_name = st.selectbox('Select Dataset', dataset_names , placeholder='Select a dataset', key='dataset_name' , help='Select the dataset to get the result')

# wait for the user to select the dataset
if dataset_name == None:
    st.stop()


# from the selected table get the metric_no columns
db,cursor = db_connect()
cursor.execute(f"SELECT distinct metric_no FROM {dataset_name}")
metrics = [metric_no[0] for metric_no in cursor.fetchall()]
db.close()

metrics.insert(0, None)


metric_no = st.selectbox('Select Metric No', metrics ,placeholder='Select a metric' , key='metric_no' , help='Select the metric number to get the result')

# wait for the user to select the metric_no
if metric_no == None:
    st.stop()

# A button to display the result
db , cursor = db_connect()
cursor.execute(f"SELECT result FROM {dataset_name} WHERE metric_no = '{metric_no}'")
result = cursor.fetchone()[0]
db.close()

if not result or result == None:
    st.subheader('No result found')

result = result.lower()

final_result = ""

if dataset_name == "background":
    if result == "got":
        final_result = "Student will get graduate"
    else:
        final_result = "Student will not graduate on time"

elif dataset_name == "employeement":
    if result == "employed":
        final_result = "Student will get the campus placement"
    else:
        final_result = "Student will not get the campus placement"

elif dataset_name == "result":
    if result == "pass":
        final_result = "The student will not attire"
    else:
        final_result = "The student will attired"


st.subheader(final_result)
