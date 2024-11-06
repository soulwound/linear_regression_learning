from matplotlib import pyplot as plt
import numpy as np
import random
import utils
features = np.array([1,2,3,5,6,7])
labels = np.array([155, 197, 244, 356, 407, 448])

utils.plot_points(features, labels)

# квадратный корень из квадратической ошибки
def rmse(labels, predictions):
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0/n * (np.dot(differences, differences)))

def square_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + num_rooms*price_per_room
    base_price += learning_rate*(price-predicted_price)
    price_per_room += learning_rate*num_rooms*(price-predicted_price)
    return price_per_room, base_price

def absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + num_rooms*price_per_room
    if price > predicted_price:
        price_per_room += learning_rate*price_per_room
        base_price += learning_rate
    else:
        price_per_room -= learning_rate*price_per_room
        base_price -= learning_rate
    return price_per_room, base_price

def linear_regression(features, labels, learning_rate=0.01, epochs = 1000):
    price_per_room = random.random()
    base_price = random.random()
    errors = []
    for epoch in range(epochs):
        #if True:
        #    utils.draw_line(price_per_room, base_price, starting=0, ending=8)
        predictions = features[0] * price_per_room + base_price
        errors.append(rmse(labels, predictions))
        i = random.randint(0, len(features)-1)
        num_rooms = features[i]
        price = labels[i]
        price_per_room, base_price = square_trick(base_price, price_per_room, num_rooms, price, learning_rate)
    utils.draw_line(price_per_room, base_price, 'black', starting=0, ending=8)
    print('Price per room:', price_per_room)
    print('Base price:', base_price)
    plt.show()
    plt.scatter(range(len(errors)), errors)
    plt.show()
    return price_per_room, base_price

plt.ylim(0,500)
linear_regression(features, labels, learning_rate = 0.01, epochs = 10000)
plt.show()