import streamlit as st
import pandas as pd
import pickle

# Define the mappings for feature conversion
mapping = {
    'Device Class': {'Low End': 0, 'Mid End': 1, 'High End': 2},
    'Location': {'Jakarta': 1, 'Bandung': 0},
    'Games Product': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Music Product': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Education Product': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Call Center': {'No': 0, 'Yes': 1},
    'Video Product': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Use MyApp': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Payment Method': {'Pulsa': 0, 'Debit': 1, 'Credit': 2, 'Digital Wallet': 3},
    # Add mappings for other features...
}

# Your mapping payment methoddictionary
mapping_payment_method_dict = {
    'Payment Method': {
        'Pulsa': 0,
        'Debit': 1,
        'Credit': 2,
        'Digital Wallet': 3
    }
}

def create_dummy_columns(df, mapping_dict):
    for method in mapping_dict['Payment Method']:
        col_name = f'Payment Method_{method}'
        df[col_name] = 0  # Initialize with 0
        
        df.loc[df['Payment Method'] == method, col_name] = 1

    return df

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

desired_col = ['Tenure Months', 'Location', 'Device Class', 'Games Product',
       'Music Product', 'Education Product', 'Call Center',
       'Video Product', 'Use MyApp', 'Monthly Purchase (Thou. IDR)',
       'Payment Method_Credit', 'Payment Method_Debit',
       'Payment Method_Digital Wallet', 'Payment Method_Pulsa']


# Load your pre-trained model
with open('churn_model_prec.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Churn Prediction App')

# File prediction
uploaded_file = st.file_uploader("Upload your csv file here...", type=['csv'])

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write("Dataframe sample:", dataframe.head())

    # Data test preparation

    dataset = dataframe[['Tenure Months', 'Location', 'Device Class', 'Games Product',
       'Music Product', 'Education Product', 'Call Center',
       'Video Product', 'Use MyApp', 'Monthly Purchase (Thou. IDR)', 'Payment Method']]
    dataset = dataset.replace(mapping)
    dataset = create_dummy_columns(dataset, mapping_payment_method_dict)
    dataset = dataset[desired_col]

    result = dataframe.copy()
    result['Churn Label(Predicted)'] = model.predict(dataset)
    result.loc[result['Churn Label(Predicted)'] == 1, 'Churn Label(Predicted)'] = 'Churn'
    result.loc[result['Churn Label(Predicted)'] == 0, 'Churn Label(Predicted)'] = 'No Churn'

    result['Churn Prediction Proba'] = model.predict_proba(dataset)[:,1]

    st.write("Prediction result sample:", result.head())


    csv = convert_df(result)

    if st.download_button("Press to Download Prediction Result", csv, 'churn_prediction_result.csv', "text/csv",key='download-csv'):
        st.write('Thanks for downloading!')


st.title('Predict Specific Customer')
st.write('Enter Customer Details:')

selected_features = list(mapping.keys()) + ['Tenure Months', 'Monthly Purchase (Thou. IDR)'] # Retrieve all available features

input_data = {}

for feature in selected_features:
    if feature == 'Tenure Months':
        user_input = st.number_input(f'{feature}', min_value=0, value=0)
        if user_input:
            input_data[feature] = user_input
    elif feature == 'Monthly Purchase (Thou. IDR)':
        user_input = st.number_input(f'{feature}', min_value=0.0, value=0.0)
        if user_input:
            input_data[feature] = user_input
    elif feature == 'Payment Method':
        user_selected_value = st.selectbox(f'{feature}', [None] + list(mapping[feature].keys()))
        if user_selected_value:
            input_data[feature] = mapping[feature][user_selected_value]

            # Enable dummies based on the chosen Payment Method
            for method in mapping[feature].keys():
                if method != user_selected_value:
                    input_data[f'Payment Method_{method}'] = 0
                else:
                    input_data[f'Payment Method_{method}'] = 1
    else:
        user_selected_value = st.selectbox(f'{feature}', [None] + list(mapping[feature].keys()))
        if user_selected_value:
            input_data[feature] = mapping[feature][user_selected_value]

if 'Payment Method' in input_data:
    input_data.pop('Payment Method')
# Now 'input_data' contains the processed numerical values for all selected features

# Convert the processed data to a DataFrame
input_df = pd.DataFrame([input_data])


if st.button('Predict'):
    if len(input_data) == len(model.feature_names_in_):  # Check if all features have been selected
        # Perform prediction here
        input_df = input_df[desired_col]
        prediction = model.predict(input_df) # Assuming 'model' is your trained model
        prediction = 'Churn' if int(prediction) == 1 else 'No Churn'

        predicted_proba = model.predict_proba(input_df)  # Get predicted probabilities
        
        st.success('The prediction result is: {}'.format(prediction))  # Show the prediction result

        # Show predicted probabilities as text
        # st.write('Predicted Probabilities:')
        # for class_, proba in zip(model.classes_, predicted_proba[0]):
        #     class_ = 'Churn' if int(class_) == 1 else 'No Churn'
        st.write(f'Probability of Churn: {predicted_proba[0][1]:.4f}')  # Display the probability for each class
    else:
        st.warning('Please select values for all features.')