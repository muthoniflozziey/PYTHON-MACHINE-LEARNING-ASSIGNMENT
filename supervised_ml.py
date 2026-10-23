from sklearn.linear_model import LinearRegression
# training data
x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]
# create the model
model = LinearRegression()
#train the model
model.fit(x, y)
# use float
hours = float(input("enter hours studied:"))
#predict
prediction = model.predict([[hours]])
# display result
print(f"predicted exam score: {prediction[0]:.2f}")
import matplotlib.pyplot as plt
# Create a graph
plt.scatter(x, y, color="blue", label="actual exam scores")
plt.plot(x, model.predict(x), color="red", label="regression line")
plt.xlabel("hours studied")
plt.ylabel("exam score")
plt.title("student exam score prediction")
plt.legend()
plt.grid(True)
plt.show()
