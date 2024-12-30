import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import psycopg2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog, QLabel
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
import sys

# Database configuration
DB_CONFIG = {
    'dbname': 'demo',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
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

class RidgeRegressionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Database Data Prediction (Ridge Regression)")
        self.setGeometry(100, 100, 500, 300)
        self.setStyleSheet("background-color: #f0f4f7;")

        layout = QtWidgets.QVBoxLayout()

        title_label = QLabel("Ridge Regression Prediction App")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title_label)

        self.fetch_training_button = QtWidgets.QPushButton("Fetch Training Data")
        self.fetch_training_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px; font-size: 14px;")
        self.fetch_training_button.clicked.connect(self.upload_training_files)
        layout.addWidget(self.fetch_training_button)

        self.fetch_test_button = QtWidgets.QPushButton("Fetch Test Data")
        self.fetch_test_button.setStyleSheet("background-color: #e67e22; color: white; padding: 10px; font-size: 14px;")
        self.fetch_test_button.clicked.connect(self.upload_test_file)
        layout.addWidget(self.fetch_test_button)

        self.setLayout(layout)

    def upload_training_files(self):
        try:
            place_table, ok1 = QInputDialog.getText(self, "Input", "Enter the name of the place timing table:")
            route_table, ok2 = QInputDialog.getText(self, "Input", "Enter the name of the route timing table:")

            if not (ok1 and ok2) or not (place_table and route_table):
                QMessageBox.warning(self, "Error", "You must provide both table names.")
                return

            global place_data, route_data
            place_data = fetch_data_from_db(place_table)
            route_data = fetch_data_from_db(route_table)

            self.prepare_and_train_model(place_data, route_data)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch training data: {e}")

    def prepare_and_train_model(self, place_data, route_data):
        try:
            place_data_sorted = place_data.sort_values(by='endpoint')
            route_data_sorted = route_data.sort_values(by='endpoint')

            selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength']
            target_column = 'slack'

            features = place_data_sorted[selected_columns]
            target = route_data_sorted[target_column]

            global scaler, ridge_model
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_train, y_train)

            y_pred = ridge_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            QMessageBox.information(self, "Success", f"Model trained successfully.\nR-squared Score: {r2:.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model training failed: {e}")

    def upload_test_file(self):
        try:
            test_table, ok = QInputDialog.getText(self, "Input", "Enter the name of the test timing table:")

            if not ok or not test_table:
                QMessageBox.warning(self, "Error", "You must provide a test table name.")
                return

            test_data = fetch_data_from_db(test_table)
            self.predict_and_save(test_data)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch test data: {e}")

    def predict_and_save(self, df):
        try:
            selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength']

            if not set(selected_columns).issubset(df.columns):
                raise ValueError(f"Uploaded table must contain the following columns: {selected_columns}")

            features = df[selected_columns].astype(float)
            features_scaled = scaler.transform(features)
            predictions = ridge_model.predict(features_scaled)

            df['predicted_route_slack'] = predictions
            output_filepath = QFileDialog.getSaveFileName(self, "Save Predictions", "predicted_route_slack.csv", "CSV Files (*.csv)")[0]

            if output_filepath:
                df[['beginpoint', 'endpoint', 'predicted_route_slack']].to_csv(output_filepath, index=False)
                QMessageBox.information(self, "Success", f"Predictions saved to {output_filepath}")
            else:
                QMessageBox.warning(self, "Warning", "Save operation canceled.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = RidgeRegressionApp()
    main_window.show()
    sys.exit(app.exec_())
