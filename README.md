# Employye_prediction

This code implements a Streamlit-based web application for predicting employee income levels using a machine learning model. Here's a brief overview of its key functionalities:

🔧 Technologies Used
Python

Streamlit: For creating the web interface

Pandas: For data handling

Matplotlib: For visualizations

Scikit-learn: For machine learning

🏗️ App Structure
1. User Interface Setup
Configures the page title and layout.

Adds a sidebar with:

Navigation between "Dashboard" and "Prediction"

File uploader to upload adult.csv dataset.

📊 Data Processing & Cleaning
Replaces missing values (?) in workclass and occupation with "Others".

Removes entries with invalid work classes like 'Without-pay', 'Never-worked'.

Filters out age and educational number outliers.

Drops redundant column education.

Encodes categorical features using LabelEncoder.

📈 Dashboard Section
When the user selects "Dashboard":

Shows charts and tables:

Age Distribution (bar chart)

Income Distribution (bar chart)

Workclass Distribution (pie chart)

Occupation Distribution (pie chart)

🤖 Prediction Section
When the user selects "Prediction":

Splits the dataset into train/test sets.

Trains a Random Forest Classifier to predict whether an individual's income is >50K or <=50K.

Displays the model accuracy.

Allows user to input custom feature values (like age, workclass, etc.) via a form.

Predicts and displays whether the income will be >50K or <=50K.

⚠️ Fallback Handling
If no file is uploaded, it shows a message prompting the user to upload adult.csv.

✅ Summary
This Streamlit app:

Lets users explore a salary dataset interactively.

Trains and evaluates a machine learning model.

Provides real-time income prediction for custom inputs.

Useful for HR analytics or data science learners.

