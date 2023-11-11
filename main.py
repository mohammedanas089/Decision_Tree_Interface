import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree


model = tree.DecisionTreeClassifier(max_depth=3)  # You can replace this with your trained model

@st.cache_data  # Cache the dataset for better performance
def load_data(file):
    df = pd.read_csv(file)
    df = df.fillna(0)
    return df

st.title("Prediction App")

st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.title("Select Columns and Predict")
    column_list = df.columns.tolist()
    numeric_columns = [col for col in column_list if pd.api.types.is_numeric_dtype(df[col])]
    selected_columns = st.sidebar.multiselect("Select Numeric Columns for Prediction", column_list)
    numeric_target_columns = [col for col in numeric_columns if col != selected_columns]
    categorical_target_columns = [col for col in column_list if col not in numeric_columns]
    
    target_column = st.sidebar.selectbox("Select Target Categorical Column for Prediction", categorical_target_columns)
    
    if selected_columns and target_column:
        # Display the selected columns with max and min values
        st.write("Selected Columns for Prediction:")

        # Create the pipeline with preprocessing and the decision tree model

        # Create the input dataset and target variable
        input_data = df[selected_columns]
        for column in input_data.columns:
            if column not in categorical_target_columns:
                input_data[column] = input_data[column].astype(str)

        # If the target column is numeric, scale it; otherwise, use label encoding

        keys_list = ["key1", "key2", "key3"]  # List of keys you want to add

        # Create a dictionary with empty values using a dictionary comprehension
        datalink={}
        target = df[target_column].astype(str)
        for k in input_data.select_dtypes(include=['object', 'category']).columns:
            le=LabelEncoder()
            temp=input_data[k].astype(str) 
            input_data[k]=le.fit_transform(input_data[k])
            datalink[k]={}
            for i,j in zip(temp,input_data[k]):
                datalink[k][i]=j
                
            del le
        X_train, X_test, y_train, y_test = train_test_split(input_data, target, test_size=0.2, random_state=42)


        model.fit(X_train, y_train)



        # Train the model on the entire dataset (you can change this to your preferred way of training)

        # Input for making predictions
        # Input for making predictions
        st.subheader("Make Predictions")
        input_values = {}

        for column in selected_columns:
            if column in categorical_target_columns:  # Check if the column is categorical
                unique_values = df[column].unique()
                selected_value = st.selectbox(f"Select {column}", unique_values)
                input_values[column] = selected_value
            else:
                try:
                    value = st.text_input(f"Enter {column} (Min: {df[column].min()}, Max: {df[column].max()})")
                    input_values[column] = value
                except:
                    pass

                # After model prediction
# After model prediction
        # After model prediction
        if st.button("Predict"):
            custom = []
            for column in selected_columns:
                if column in categorical_target_columns:
                    print(column,input_values[column])
                    custom.append(datalink[column][input_values[column]])  # Use the selected dropdown value
                else:
                    custom.append(input_values[column])
            print("Custom data",custom)
            pred = model.predict([custom])
        
            if target_column in categorical_target_columns:  # Check if the target column was label encoded
                label_encoder = LabelEncoder()
                all_target_values = pd.concat([df[target_column], pd.Series(pred[0])])  # Combine original values and prediction
                label_encoder.fit(all_target_values)  # Fit on all possible labels
                st.write(f"Predicted {target_column}: {pred}")
            else:
                st.write(f"Predicted {target_column}: {pred}")
             



        # Display feature importances (Note: Feature importances may not be directly interpretable in this case)
        st.subheader("Feature Importances")
        try:
            feature_importances = model.feature_importances_

            for col, importance in zip(selected_columns, feature_importances):
                st.write(f"{col}: {importance}")
        except AttributeError:
            st.write("Feature importances are not available for this model.")

        # Display the decision tree (Note: May not work well with large datasets)
        #st.subheader("Decision Tree Visualization")
        #st.write("Note: Decision tree visualization is for demonstration purposes and may not work well with large datasets.")
        #st.image("decision_tree.png", use_column_width=True)
       
        try:
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': selected_columns, 'Importance': feature_importances})
            st.bar_chart(importance_df.set_index('Feature'))
        except AttributeError:
            st.write("Feature importances are not available for this model.")

        st.subheader("Decision Tree Visualization")
        
        plt.figure(figsize=(15, 10))
        tree.plot_tree(model, feature_names=list(input_data.columns), filled=True)
        st.pyplot()
        # Visualize the first tree in the forest (you can change the index)
            
       
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        st.subheader("Confusion Matrix")
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d')
            st.pyplot()
        except AttributeError:
            st.write("Confusion matrix is not available for this model.")


        #import pdpbox

        #st.subheader("Partial Dependence Plot")
        
        #pdp_feature = selected_columns[0]  # Choose a feature to plot
        #st.write(pdp_feature)
        #pdp_dist = pdpbox.pdp.PDPIsolate(model=model, df=X_train, model_features=selected_columns, feature=pdp_feature,feature_name="Something")
        #pdpbox.pdp_plot(pdp_dist, pdp_feature)
        #st.pyplot()




    else:
        st.warning("Please select columns for prediction and a target column in the sidebar.")
