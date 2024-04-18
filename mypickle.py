import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Read the data
df = pd.read_csv('amazon.csv')

# Split the data into features and target variable
X = df.drop(columns=['Weight'])  # Features
y = df['Weight']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Random Forest Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('amazon_regression.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Regression model saved successfully as 'amazon_regression.pkl'")
