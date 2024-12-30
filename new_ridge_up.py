import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Treeview
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv("place_timing_cleaned.csv")
data1 = pd.read_csv("route_timing_cleaned.csv")

# Sort both tables by 'beginpoint'
data_sorted = data.sort_values(by='endpoint')
data1_sorted = data1.sort_values(by='endpoint')

selected_columns = ['fanout', 'netcount', 'netdelay', 'invdelay', 'bufdelay', 'seqdelay', 'skew', 'combodelay', 'wirelength','slack']  # Features
target_column = 'slack'  # Target

# Prepare features and target
features = data_sorted[selected_columns]
target = data1_sorted[target_column]

# Standardize the features (Ridge regression is sensitive to feature scaling)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0, random_state=42)  # alpha controls regularization strength
ridge_model.fit(X_train, y_train)

# Evaluate model
y_pred = ridge_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R-squared Score:", r2)
print("Training Score:", ridge_model.score(X_train, y_train))
print("Test Score:", ridge_model.score(X_test, y_test))

# GUI Implementation
def upload_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not filepath:
        return
    try:
        test_data = pd.read_csv(filepath)
        predict_and_save(test_data)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file: {e}")

def predict_and_save(df):
    try:
        # Ensure required columns exist
        if not set(selected_columns).issubset(df.columns):
            raise ValueError(f"Uploaded file must contain the following columns: {selected_columns}")

        # Extract features, scale them, and predict
        features = df[selected_columns].astype(float)
        features_scaled = scaler.transform(features)
        predictions = ridge_model.predict(features_scaled)

        # Add predictions to the DataFrame
        df['predicted_route_slack'] = predictions

        # Save to a new CSV file in the current directory
        output_filepath = "predicted_route_slack.csv"
        df[['beginpoint', 'endpoint', 'predicted_route_slack']].to_csv(output_filepath, index=False)

        messagebox.showinfo("Success", f"Predictions saved to {output_filepath} in the current directory")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# Create GUI window
root = tk.Tk()
root.title("CSV File Prediction (Ridge Regression)")

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

upload_button = tk.Button(frame, text="Upload CSV", command=upload_file)
upload_button.pack(side=tk.TOP, pady=5)

root.mainloop()
