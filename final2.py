import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import psycopg2
import sys

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

class DataProcessingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Data Cleaning and Ridge Regression App")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #f0f4f7;")

        layout = QtWidgets.QVBoxLayout()

        title_label = QLabel("Data Cleaning and Ridge Regression App")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title_label)

        self.cleaning_button = QtWidgets.QPushButton("Fetch and Clean Data from Database")
        self.cleaning_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px; font-size: 14px;")
        self.cleaning_button.clicked.connect(self.clean_data_from_db)
        layout.addWidget(self.cleaning_button)

        self.train_button = QtWidgets.QPushButton("Train Model")
        self.train_button.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-size: 14px;")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        self.test_button = QtWidgets.QPushButton("Test Model")
        self.test_button.setStyleSheet("background-color: #e67e22; color: white; padding: 10px; font-size: 14px;")
        self.test_button.clicked.connect(self.test_model)
        self.test_button.setEnabled(False)
        layout.addWidget(self.test_button)

        self.setLayout(layout)

    def clean_data_from_db(self):
        try:
            place_table, ok1 = QInputDialog.getText(self, "Input", "Enter the name of the place timing table:")
            route_table, ok2 = QInputDialog.getText(self, "Input", "Enter the name of the route timing table:")

            if not (ok1 and ok2) or not (place_table and route_table):
                QMessageBox.warning(self, "Error", "You must provide both table names.")
                return

            place_data = fetch_data_from_db(place_table)
            route_data = fetch_data_from_db(route_table)

            if 'endpoint' not in place_data.columns or 'endpoint' not in route_data.columns:
                QMessageBox.critical(self, "Error", "Both tables must contain the 'endpoint' column.")
                return

            place_data_sorted = place_data.sort_values(by='endpoint')
            route_data_sorted = route_data.sort_values(by='endpoint')

            place_data_filtered = place_data_sorted[place_data_sorted['endpoint'].isin(route_data_sorted['endpoint'])]
            route_data_filtered = route_data_sorted[route_data_sorted['endpoint'].isin(place_data_sorted['endpoint'])]

            def clean_data(df):
                df_cleaned = df.dropna()
                df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: row.astype(str).str.contains(r'\bna\b', case=False)).any(axis=1)]
                return df_cleaned

            self.cleaned_place_data = clean_data(place_data_filtered)
            self.cleaned_route_data = clean_data(route_data_filtered)

            QMessageBox.information(self, "Success", "Data cleaned successfully.")
            self.train_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean data: {e}")

    def train_model(self):
        try:
            selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength']
            target_column = 'slack'

            features = self.cleaned_place_data[selected_columns]
            target = self.cleaned_route_data[target_column]

            global scaler, ridge_model
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_train, y_train)

            y_pred = ridge_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            QMessageBox.information(self, "Success", f"Model trained successfully.\nR-squared Score: {r2:.2f}")
            self.test_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model training failed: {e}")

    def test_model(self):
        try:
            test_table, ok = QInputDialog.getText(self, "Input", "Enter the name of the test timing table:")

            if not ok or not test_table:
                QMessageBox.warning(self, "Error", "You must provide a test table name.")
                return

            test_data = fetch_data_from_db(test_table)

            selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength']

            if not set(selected_columns).issubset(test_data.columns):
                raise ValueError(f"Test table must contain the following columns: {selected_columns}")

            features = test_data[selected_columns].astype(float)
            features_scaled = scaler.transform(features)
            predictions = ridge_model.predict(features_scaled)

            test_data['predicted_route_slack'] = predictions
            output_file, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "predicted_route_slack.csv", "CSV Files (*.csv)")

            if output_file:
                test_data[['beginpoint', 'endpoint', 'predicted_route_slack']].to_csv(output_file, index=False)
                QMessageBox.information(self, "Success", f"Predictions saved to {output_file}")
            else:
                QMessageBox.warning(self, "Warning", "Save operation canceled.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = DataProcessingApp()
    main_window.show()
    sys.exit(app.exec_())
