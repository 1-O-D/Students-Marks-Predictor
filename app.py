import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Student Marks Predictor")
st.write("### Machine Learning Mini Project")
st.write("Predict a student's marks based on study hours using Linear Regression.")

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("student_marks.csv")

X = data[["Hours"]]
y = data["Marks"]

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# User Input
# -------------------------------
hours = st.slider(
    "Select Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Marks"):

    prediction = model.predict([[hours]])[0]

    st.success(f"📈 Predicted Marks: **{prediction:.2f}**")

# -------------------------------
# Show Dataset
# -------------------------------
with st.expander("📄 View Dataset"):
    st.dataframe(data)

# -------------------------------
# Graph
# -------------------------------
fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(X, y, color="blue", label="Actual Data")
ax.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")

ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
ax.set_title("Student Marks Prediction")

ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.write("Made with ❤️ using Streamlit & Scikit-Learn")
