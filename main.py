# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Create dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Sleep_Hours": [7,6,8,5,7,6,8,5,7,6],
    "Previous_Score": [50,55,60,65,70,75,80,85,90,95],
    "Final_Score": [52,57,63,68,72,78,83,88,92,97]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# Step 3: Split input and output
X = df[["Hours_Studied", "Sleep_Hours", "Previous_Score"]]
y = df["Final_Score"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate
score = r2_score(y_test, y_pred)
print("Model Accuracy (R2 Score):", score)

# Step 8: Predict new value
new_data = pd.DataFrame([[6, 7, 80]], columns=["Hours_Studied", "Sleep_Hours", "Previous_Score"])  # hours studied, sleep, previous score
prediction = model.predict(new_data)

print("Predicted Final Score:", prediction[0])
plt.figure(figsize=(6, 6))
y_all_pred = model.predict(X)
plt.scatter(y, y_all_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted")
plt.show()