import turicreate as tc
import numpy as np
import matplotlib.pyplot as plt
import utils

data = tc.SFrame('Hyderabad.csv')
model = tc.linear_regression.create(data, target='Price')
house = tc.SFrame({'Area': [1000], 'No. of Bedrooms':[3]})
#print(model.predict(house))
model.predict(house)

plt.scatter(data['Area'], data['Price'])