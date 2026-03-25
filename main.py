import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data.csv")

# Input and output
X = data[['Sleep', 'Study', 'Screen', 'Mood']]
y = data['Productivity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Custom test
sample = [[7, 4, 3, 2]]
prediction = model.predict(sample)

print("Predicted Productivity:", prediction)