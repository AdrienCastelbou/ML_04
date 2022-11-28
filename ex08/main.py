from my_logistic_regression import MyLogisticRegression as mylogr
import numpy as np

theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# Example 1:
model1 = mylogr(theta, lambda_=5.0)
print(model1.penality)

print(model1.lambda_)

# Example 2:
model2 = mylogr(theta, penality=None)
print(model2.penality)
# Output
None
print(model2.lambda_)

# Example 3:
model3 = mylogr(theta, penality=None, lambda_=2.0)
print(model3.penality)
# Output
None
print(model3.lambda_)
# Output
0.0