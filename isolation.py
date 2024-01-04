import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('UNSW_NB15.csv')  # Replace with the actual path to your dataset

# Preprocess the dataset

# Handling missing values (you can adjust this as needed)
df = df.dropna()

# Label encode only the categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['proto', 'service', 'state', 'attack_cat']  # Add other categorical columns as needed
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Define features (X) and labels (y)
X = df.drop(columns=['label', 'id'])  # Adjust column names as needed
y = df['label']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)

# Fit the model to the training data
model.fit(X_train)

# Predict anomalies on the test data
y_pred = model.predict(X_test)

# Convert predictions to binary labels (1 for anomalies, 0 for inliers)
y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

# Evaluate the model's performance
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_binary)  # Use y_pred_binary instead of y_pred
print(f"\nROC AUC Score: {roc_auc:.2f}")

# Create a graphical representation of the results (plotting inliers and outliers)
inliers = X_test[y_pred == 1]
outliers = X_test[y_pred == -1]

plt.figure(figsize=(10, 6))
plt.scatter(inliers.iloc[:, 0], inliers.iloc[:, 1], label='Inliers', s=5)
plt.scatter(outliers.iloc[:, 0], outliers.iloc[:, 1], label='Outliers', s=5, color='red')
plt.legend()
plt.title ('Isolation Forest - Inliers and Outliers')
plt.xlabel('Feature 1')  # Adjust the feature names as needed
plt.ylabel('Feature 2')  # Adjust the feature names as needed
plt.show()
