# Import necessary libraries
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
# Replace this with the correct path to your dataset
dataset_path = 'Financial_inclusion_dataset.csv'
data = pd.read_csv(dataset_path)

# Step 2: Preprocess the dataset
# Initialize LabelEncoders for categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    if col != 'uniqueid':  # Exclude the unique ID column
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Step 3: Split the data into features (X) and target (y)
X = data.drop(['bank_account', 'uniqueid'], axis=1)
y = data['bank_account']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Save the trained model
model_path = "random_forest_model.pkl"
joblib.dump(clf, model_path)

# Step 7: Load the trained model for predictions
clf = joblib.load(model_path)

# Step 8: Create mappings for categorical columns
# Store mappings for human-readable values and their encoded counterparts
original_label_mappings = {}
for col, le in label_encoders.items():
    original_label_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))

# Streamlit app starts here
st.title("Financial Inclusion Prediction App")
st.markdown("Enter the demographic details to predict financial inclusion.")

# Step 9: Define the input columns and their types
input_columns = [
    ("country", "categorical"),
    ("year", "numerical"),
    ("location_type", "categorical"),
    ("cellphone_access", "categorical"),
    ("household_size", "numerical"),
    ("age_of_respondent", "numerical"),
    ("gender_of_respondent", "categorical"),
    ("relationship_with_head", "categorical"),
    ("marital_status", "categorical"),
    ("education_level", "categorical"),
    ("job_type", "categorical"),
]

# Step 10: Create a dictionary to store user inputs
user_input = {}

# Step 11: Loop through each column and dynamically generate input fields
for col, col_type in input_columns:
    if col_type == "categorical":
        # Use st.selectbox for categorical columns
        user_input[col] = st.selectbox(
            col.replace("_", " ").title(),  # Make column name readable
            options=original_label_mappings[col].keys()  # Display human-readable labels
        )
    elif col_type == "numerical":
        # Use st.number_input for numerical columns
        user_input[col] = st.number_input(
            col.replace("_", " ").title(),  # Make column name readable
            min_value=int(data[col].min()),  # Minimum value from the dataset
            max_value=int(data[col].max()),  # Maximum value from the dataset
            step=1,  # Increment step
        )

# Step 12: Convert categorical inputs to encoded values
for col, col_type in input_columns:
    if col_type == "categorical":
        # Map human-readable labels to their encoded values
        user_input[col] = original_label_mappings[col][user_input[col]]

# Step 13: Convert the user input dictionary to a DataFrame
# This ensures compatibility with the machine learning model
user_input_df = pd.DataFrame([user_input])

# Step 14: Prediction and Display
if st.button("Predict"):
    # Perform prediction using the trained model
    prediction = clf.predict(user_input_df)
    
    # Interpret the prediction result
    result = "Has a Bank Account" if prediction[0] == 1 else "Does Not Have a Bank Account"
    
    # Display the result to the user
    st.write(f"Prediction: {result}")
