# streamlit_app.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Employee Salary Prediction App", layout="wide")
st.title("Employee Salary Prediction App 💼📊")
st.markdown("""
<style>
.main .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)


# Sidebar for navigation and file upload
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])
uploaded_file = st.sidebar.file_uploader("Upload the 'adult.csv' file", type="csv")



if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Cleaning
    data['workclass'].replace({'?': 'Others'}, inplace=True)
    data['occupation'].replace({'?': 'Others'}, inplace=True)
    data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]

    # Outlier removal
    data = data[(data['age'] >= 17) & (data['age'] <= 75)]
    data = data[(data['educational-num'] >= 5) & (data['educational-num'] <= 16)]

    # Drop redundant feature
    data = data.drop(columns=['education'])

    # Label Encoding
    encoder = LabelEncoder()
    categorical_cols = ['workclass', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country']
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])

    # Dashboard Section
    if section == "Dashboard":
        st.markdown("---")
        st.subheader("Data Visualizations")
        chart_option = st.selectbox(
            "Select Distribution to View in Detail",
            ["Age Distribution (Bar Chart)", "Income Distribution (Bar Chart)", "Workclass Distribution (Pie Chart)", "Occupation Distribution (Pie Chart)"]
        )

        if chart_option == "Age Distribution (Bar Chart)":
            st.write("### Age Distribution (Bar Chart)")
            st.bar_chart(data['age'].value_counts().sort_index(), use_container_width=False)
            st.write("**Details:**")
            st.dataframe(data['age'].value_counts().sort_index().reset_index().rename(columns={"index": "Age", "age": "Count"}))

        elif chart_option == "Income Distribution (Bar Chart)":
            st.write("### Income Distribution (Bar Chart)")
            st.bar_chart(data['income'].value_counts(), use_container_width=False)
            st.write("**Details:**")
            st.dataframe(data['income'].value_counts().reset_index().rename(columns={"index": "Income", "income": "Count"}))

        elif chart_option == "Workclass Distribution (Pie Chart)":
            st.write("### Workclass Distribution (Pie Chart)")
            workclass_counts = data['workclass'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(3,3))
            ax1.pie(workclass_counts, labels=workclass_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1, use_container_width=False)
            st.write("**Details:**")
            st.dataframe(workclass_counts.reset_index().rename(columns={"index": "Workclass", "workclass": "Count"}))

        elif chart_option == "Occupation Distribution (Pie Chart)":
            st.write("### Occupation Distribution (Pie Chart)")
            occupation_counts = data['occupation'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(3,3))
            ax2.pie(occupation_counts, labels=occupation_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2, use_container_width=False)
            st.write("**Details:**")
            st.dataframe(occupation_counts.reset_index().rename(columns={"index": "Occupation", "occupation": "Count"}))

    # Prediction Section
    if section == "Prediction":
        x = data.drop(columns=['income'])
        y = data['income'].apply(lambda x: 1 if x.strip() == ">50K" else 0)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        # Train model
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

        st.markdown("---")
        st.subheader("Try a Prediction")
        with st.form("prediction_form"):
            user_input = {col: st.number_input(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
                          for col in x.columns}
            submitted = st.form_submit_button("Predict Income")
        if submitted:
            user_df = pd.DataFrame([user_input])
            prediction = model.predict(user_df)[0]
            st.info("Predicted: Income >50K" if prediction == 1 else "Predicted: Income <=50K")

else:
    st.info("Please upload the 'adult.csv' file from the sidebar to get started.")

    # Split data
    x = data.drop(columns=['income'])
    y = data['income'].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Predict on user input
    st.subheader("Try a Prediction")
    user_input = {col: st.number_input(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
                  for col in x.columns}
    user_df = pd.DataFrame([user_input])

    if st.button("Predict Income"):
        prediction = model.predict(user_df)[0]
        st.info("Predicted: Income >50K" if prediction == 1 else "Predicted: Income <=50K")
