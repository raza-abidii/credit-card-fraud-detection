import streamlit as st
import pandas as pd
from models import predict_fraud

def main():
    st.title("Credit Card Fraud Detection")
    
    # Create input form using Streamlit widgets
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0)
        city_population = st.number_input("City Population", min_value=0)
    
    with col2:
        merchant_latitude = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0)
        merchant_longitude = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0)
        category = st.text_input("Merchant Category")
    
    if st.button("Check Transaction"):
        if not all([amount, latitude, longitude, city_population, merchant_latitude, merchant_longitude, category]):
            st.warning("Please fill in all fields")
        else:
            try:
                # Create input DataFrame
                input_data = pd.DataFrame({
                    'amt': [float(amount)],
                    'lat': [float(latitude)],
                    'long': [float(longitude)],
                    'city_pop': [float(city_population)],
                    'merch_lat': [float(merchant_latitude)],
                    'merch_long': [float(merchant_longitude)],
                    'category': [str(category)]
                })
                
                # Make prediction
                prediction = predict_fraud(input_data)
                
                # Display result with styling
                if prediction:
                    st.error("⚠️ Fraud Detected! This transaction appears to be fraudulent.")
                else:
                    st.success("✅ Transaction appears to be legitimate.")
                
                # Display transaction details
                st.subheader("Transaction Details")
                details = pd.DataFrame({
                    'Feature': ['Amount', 'Location', 'City Population', 'Merchant Location', 'Category'],
                    'Value': [
                        f"${amount:.2f}",
                        f"({latitude}, {longitude})",
                        f"{city_population:,}",
                        f"({merchant_latitude}, {merchant_longitude})",
                        category
                    ]
                })
                st.table(details)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # Add file uploader for batch processing
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV file for batch processing", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            required_columns = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Make predictions for each row
                predictions = []
                for _, row in df.iterrows():
                    input_data = pd.DataFrame([row])
                    pred = predict_fraud(input_data)
                    predictions.append(pred)
                
                # Add predictions to dataframe
                df['prediction'] = predictions
                
                # Display summary
                fraud_count = sum(predictions)
                total_count = len(predictions)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", total_count)
                with col2:
                    st.metric("Fraudulent Transactions", fraud_count)
                with col3:
                    st.metric("Fraud Percentage", f"{(fraud_count/total_count)*100:.2f}%")
                
                # Display results table
                st.subheader("Detailed Results")
                st.dataframe(df)
                
                # Add download button for results
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