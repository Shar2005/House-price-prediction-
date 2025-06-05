import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the Data
# Load the Boston Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Display the first few rows of the dataset
# print(df.head()) for test
df.head()

# Explore the Data
# Summary statistics
# print(df.describe()) for test
df.describe()

# Info about the dataset
# print(df.info()) for test
df.info()

# Checking for missing values
# print(df.isnull().sum())
df.isnull().sum()

# Data Preprocessing
# Feature Selection and Data Splitting
# Defining features and target variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
# Training a Linear Regression Model
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R² Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error : {mse}")
print(f"R² Score: {r2}")

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()
