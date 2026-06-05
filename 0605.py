import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_and_plot(n=200):
    # Randomly choose parameters within the specified ranges
    a = np.random.uniform(-50, 50)
    b = np.random.uniform(0, 100)
    var = np.random.uniform(0, 300)
    
    # Generate n points
    x = np.random.uniform(-100, 100, n)
    
    # N(0, var) -> variance is var, so standard deviation is sqrt(var)
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
    
    print(f"True a: {a:.2f}, True b: {b:.2f}, True var: {var:.2f}")
    print(f"Estimated a: {a_best:.2f}, Estimated b: {b_best:.2f}")
    
    # Calculate residuals to find outliers
    y_pred = model.predict(x_reshaped)
    residuals = np.abs(y - y_pred)
    
    # Find top 10 outliers (points with largest absolute residuals)
    outlier_indices = np.argsort(residuals)[-10:]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot all normal points
    plt.scatter(x, y, color='blue', label='Data Points', alpha=0.6)
    
    # Highlight top 10 outliers
    plt.scatter(x[outlier_indices], y[outlier_indices], color='orange', 
                label='Top 10 Outliers', edgecolor='black', s=100)
    
    # Plot regression line
    x_line = np.linspace(-100, 100, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='red', label='Regression Line (Red)', linewidth=2)
    
    plt.title('Linear Regression with Top 10 Outliers')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # You can change the number of points here
    generate_and_plot(n=200)
