import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Linear Regression & Outlier Detection")
st.write("Adjust the parameters in the sidebar to generate new data and see the linear regression update in real-time.")

# Sidebar controls
st.sidebar.header("Parameters")
n = st.sidebar.slider("Number of points (n)", min_value=10, max_value=1000, value=200, step=10)
a = st.sidebar.slider("Slope (a)", min_value=-50.0, max_value=50.0, value=10.0, step=1.0)
b = st.sidebar.slider("Intercept (b)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
var = st.sidebar.slider("Variance (var)", min_value=0.0, max_value=300.0, value=100.0, step=5.0)

# Button to manually refresh randomness
st.sidebar.button("Regenerate Random Noise")

# Generate data
x = np.random.uniform(-100, 100, n)
std_dev = np.sqrt(var)
noise = np.random.normal(0, std_dev, n)
y = a * x + b + noise

# Reshape x for sklearn
x_reshaped = x.reshape(-1, 1)

# Apply linear regression
model = LinearRegression()
model.fit(x_reshaped, y)

a_best = model.coef_[0]
b_best = model.intercept_

# Display metrics
col1, col2 = st.columns(2)
with col1:
    st.info(f"**True Parameters**\n\na = {a:.2f}\nb = {b:.2f}\nvar = {var:.2f}")
with col2:
    st.success(f"**Estimated Parameters**\n\na = {a_best:.2f}\nb = {b_best:.2f}")

# Calculate residuals to find outliers
y_pred = model.predict(x_reshaped)
residuals = np.abs(y - y_pred)

# Find top 10 outliers
# If n < 10, just highlight all points
num_outliers = min(10, n)
outlier_indices = np.argsort(residuals)[-num_outliers:]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot all normal points
ax.scatter(x, y, color='blue', label='Data Points', alpha=0.6)

# Highlight outliers
ax.scatter(x[outlier_indices], y[outlier_indices], color='orange', 
            label=f'Top {num_outliers} Outliers', edgecolor='black', s=100)

# Plot regression line
x_line = np.linspace(-100, 100, 100).reshape(-1, 1)
y_line = model.predict(x_line)
ax.plot(x_line, y_line, color='red', label='Regression Line', linewidth=2)

ax.set_title('Linear Regression and Outliers')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True)

# Display plot in Streamlit
st.pyplot(fig)
