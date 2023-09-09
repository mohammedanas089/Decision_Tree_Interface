import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the decision tree model
model = tree.DecisionTreeClassifier(max_depth=3)  # You can replace this with your trained model

# Load the dataset
@st.cache_data  # Cache the dataset for better performance
def load_data(file):
    df = pd.read_csv(file)
    df = df.fillna(0)
    print(df.isna().sum())
    return df

st.title("Student Performance Prediction App")

# File upload section
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df.info()
    # Sidebar
    st.sidebar.title("Select Columns and Predict")
    column_list = df.columns.tolist()
    # Filter numeric columns
    numeric_columns = [col for col in column_list if pd.api.types.is_numeric_dtype(df[col])]
    # Select columns for prediction
    selected_columns = st.sidebar.multiselect("Select Numeric Columns for Prediction", numeric_columns)
    
    # Select the target column for prediction and provide labels
    numeric_target_columns = [col for col in numeric_columns if col != selected_columns]
    categorical_target_columns = [col for col in column_list if col not in numeric_columns]
    
    target_column_type = st.sidebar.radio("Select Target Column Type", ["Numeric", "Categorical"])
    
    if target_column_type == "Numeric":
        target_column = st.sidebar.selectbox("Select Target Numeric Column for Prediction", numeric_target_columns)
    else:
        target_column = st.sidebar.selectbox("Select Target Categorical Column for Prediction", categorical_target_columns)
    
    if selected_columns and target_column:
        # Display the selected columns with max and min values
        st.write("Selected Columns for Prediction:")
        for column in selected_columns:
            col_min = df[column].min()
            col_max = df[column].max()
            st.write(f"{column} (Min: {col_min}, Max: {col_max})")

        # Create the pipeline with preprocessing and the decision tree model
        clf = Pipeline(steps=[('classifier', model)])

        # Create the input dataset and target variable
        input_data = df[selected_columns]
        
        # If the target column is numeric, scale it; otherwise, use label encoding
        if pd.api.types.is_numeric_dtype(df[target_column]):
            target = df[target_column]
        else:
            target = LabelEncoder().fit_transform(df[target_column])
        print("INPUT\t",input_data)
        print("OUTPUT\t",target)
        # Train the model on the entire dataset (you can change this to your preferred way of training)
        clf.fit(input_data, target)

        # Input for making predictions
        st.subheader("Make Predictions")
        input_values = {}

        for column in selected_columns:
            value = st.text_input(f"Enter {column} (Min: {df[column].min()}, Max: {df[column].max()})")
            input_values[column] = value
    

        if st.button("Predict"):
            input_df = pd.DataFrame([input_values])

            # If the target column is categorical, apply label encoding
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                input_df[target_column] = LabelEncoder().fit_transform(input_df[target_column])

            prediction = clf.predict(input_df)

            # If the target column was label encoded, decode it
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                prediction = LabelEncoder().inverse_transform(prediction)

            st.write(f"Predicted {target_column}: {prediction[0]}")

        # Display feature importances (Note: Feature importances may not be directly interpretable in this case)
        st.subheader("Feature Importances")
        try:
            feature_importances = clf.named_steps['classifier'].feature_importances_

            for col, importance in zip(selected_columns, feature_importances):
                st.write(f"{col}: {importance}")
        except AttributeError:
            st.write("Feature importances are not available for this model.")

        # Display the decision tree (Note: May not work well with large datasets)
        st.subheader("Decision Tree Visualization")
        st.write("Note: Decision tree visualization is for demonstration purposes and may not work well with large datasets.")
        st.image("decision_tree.png", use_column_width=True)

    else:
        st.warning("Please select columns for prediction and a target column in the sidebar.")
