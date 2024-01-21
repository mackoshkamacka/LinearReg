import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('data.csv')

def loss_function(m, b, points):
    total_error = 0 
    for i in range(len(points)):
        x = points.iloc[i].rows
        y = points.iloc[i].column
        total_error += (y-(m*x+b))**2
    total_error/float(len(points))


def gradient_descent(m_curr, b_curr, points, l): 
    m_grad = 0 
    b_grad = 0 
    n = len(points)

    for i in range (n):
        x = points.iloc[i].column
        y = points.iloc[i].row
        
        m_grad += (-2/n) * x * (y-(m_curr * x + b_curr))
        b_grad += (-2/n) * (y -(m_curr * x + b_curr))
        # = sum of differential of mean squared functions with respect to m and b respectfully

    m = m_curr - m_grad*l
    b = b_curr - b_grad*l
    return m,b

m = 1
b = 0
l = 0.0001 #change based on datasaet
iter = 201

for i in range(iter):
    if i %100 == 0:    
        print(f"Iteration: {i}")
        print(m)
        print(b)
    m,b = gradient_descent(m,b,data,l)

#denormalize coefficients
plt.figure(figsize=(10,6))
plt.scatter(data.row, data.column, color='blue', s=7)
plt.title("x vs. y")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color = "darkblue")
#slope, intercept,r,p,std_err = data.linregress(data.Year,data.Income)
plt.show()
