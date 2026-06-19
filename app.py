import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv("student_marks.csv")

# Input and Output
X = data[["Hours"]]
y = data["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("\n====== Student Marks Predictor ======\n")

while True:
    value = input("Enter Study Hours (q to quit): ")

    if value.lower() == "q":
        print("Goodbye!")
        break

    hours = float(value)

    prediction = model.predict([[hours]])

    print(f"Predicted Marks : {prediction[0]:.2f}\n")

# Graph
plt.scatter(X, y)
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.grid(True)
plt.show()
