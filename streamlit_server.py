import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import psycopg2
import streamlit as st

# Database configuration
DB_CONFIG = {
    'dbname': 'results',
    'user': 'postgres',
    'password': 'Welcom@123',
    'host': '172.16.16.23',
    'port': '5432',
}

def fetch_data_from_db(table_name):
    """Fetch data from a specified table in PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        raise ValueError(f"Failed to fetch data from table {table_name}: {e}")

def fetch_table_names():
    """Fetch all table names from the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['table_name'].tolist()
    except Exception as e:
        raise ValueError(f"Failed to fetch table names: {e}")

# Initialize Streamlit session state variables
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'ridge_model' not in st.session_state:
    st.session_state.ridge_model = None

# Streamlit app
st.title(" Slack Prediction")
st.sidebar.header("Options")

# Fetch table names
try:
    table_names = fetch_table_names()
except Exception as e:
    st.error(f"Error fetching table names: {e}")
    table_names = []

# Step 1: Fetch training data
st.header("Step 1: Fetch Training Data")

place_table = st.selectbox("Select the place timing table:", table_names)
route_table = st.selectbox("Select the route timing table:", table_names)

if st.button("Fetch Training Data"):
    if not place_table or not route_table:
        st.error("Please select both tables.")
    else:
        try:
            place_data = fetch_data_from_db(place_table)
            route_data = fetch_data_from_db(route_table)
            st.success("Training data fetched successfully!")
            st.write("Place Data Sample:")
            st.write(place_data.head())
            st.write("Route Data Sample:")
            st.write(route_data.head())
        except Exception as e:
            st.error(f"Error: {e}")

# Step 2: Train the model
st.header("Step 2: Train the Model")

if "place_data" in locals() and "route_data" in locals():
    selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength']
    target_column = 'slack'

    try:
        place_data_sorted = place_data.sort_values(by='endpoint')
        route_data_sorted = route_data.sort_values(by='endpoint')

        features = place_data_sorted[selected_columns]
        target = route_data_sorted[target_column]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train, y_train)

        y_pred = ridge_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # Save to session state
        st.session_state.scaler = scaler
        st.session_state.ridge_model = ridge_model

        st.success(f"Model trained successfully! R-squared Score: {r2:.2f}")
    except Exception as e:
        st.error(f"Model training failed: {e}")

# Step 3: Fetch test data and predict
st.header("Step 3: Fetch Test Data and Predict")

test_table = st.selectbox("Select the test timing table:", table_names)

if st.button("Fetch Test Data and Predict"):
    if not test_table:
        st.error("Please select the test table.")
    else:
        try:
            test_data = fetch_data_from_db(test_table)

            selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength']
            if not set(selected_columns).issubset(test_data.columns):
                st.error(f"Test table must contain the following columns: {selected_columns}")
            else:
                if st.session_state.scaler is None or st.session_state.ridge_model is None:
                    st.error("Model is not trained yet. Please train the model first.")
                else:
                    scaler = st.session_state.scaler
                    ridge_model = st.session_state.ridge_model

                    features = test_data[selected_columns].astype(float)
                    features_scaled = scaler.transform(features)
                    predictions = ridge_model.predict(features_scaled)

                    test_data['predicted_route_slack'] = predictions
                    st.success("Predictions generated successfully!")
                    st.write(test_data[['beginpoint', 'endpoint', 'predicted_route_slack']])

                    # Save predictions
                    csv = test_data[['beginpoint', 'endpoint', 'predicted_route_slack']].to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predicted_route_slack.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
