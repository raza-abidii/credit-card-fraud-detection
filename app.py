import streamlit as st
import pandas as pd
from models import predict_fraud

def main():
    st.title("Credit Card Fraud Detection")
    
    st.subheader("Single Transaction Analysis")
    with st.form(key='single_transaction_form'):
        amt = st.number_input("Transaction Amount", min_value=0.0)
        lat = st.number_input("Customer Latitude")
        long = st.number_input("Customer Longitude")
        city_pop = st.number_input("City Population", min_value=0)
        merch_lat = st.number_input("Merchant Latitude")
        merch_long = st.number_input("Merchant Longitude")
        
        category = st.selectbox("Merchant Category", options=[
            "Groceries",
            "Electronics",
            "Clothing",
            "Travel",
            "Dining",
            "Entertainment",
            "Health & Beauty",
            "Online Services",
            "Automotive",
            "Other"
        ])
        
        submit_button = st.form_submit_button(label='Check Transaction')
        
        if submit_button:
            input_data = pd.DataFrame({
                'amt': [amt],
                'lat': [lat],
                'long': [long],
                'city_pop': [city_pop],
                'merch_lat': [merch_lat],
                'merch_long': [merch_long],
                'category': [category]
            })
            prediction = predict_fraud(input_data)
            st.write(f"Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")

    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV file for batch processing", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                predictions = []
                for _, row in df.iterrows():
                    input_data = pd.DataFrame([row])
                    pred = predict_fraud(input_data)
                    predictions.append(pred)
                
                df['prediction'] = predictions
                fraud_count = sum(predictions)
                total_count = len(predictions)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", total_count)
                with col2:
                    st.metric("Fraudulent Transactions", fraud_count)
                with col3:
                    st.metric("Fraud Percentage", f"{(fraud_count/total_count)*100:.2f}%")
    
                st.subheader("Detailed Results")
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == '__main__':
    main()