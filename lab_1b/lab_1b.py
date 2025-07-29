import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

torch.set_default_dtype(torch.float64)
plt.style.use('default')

data = torch.from_numpy(np.genfromtxt('Penn-GPA-and-HS-Data.csv', delimiter=",", skip_header=1, dtype=float))

hs_gpa = data[:,1].unsqueeze(1)
sat = data[:,2].unsqueeze(1)
penn_gpa = data[:,4].unsqueeze(1)

def root_mean_squared_error(x, y, w):
    v = y - torch.matmul(x, w)
    return math.sqrt(torch.matmul(v.T, v) / len(y))

# Normalize inputs

def normalize_vector(x):
    n = len(x)
    average = (x.sum() / n)
    x_avg = average.expand(n).unsqueeze(1)

    x_centered = x - x_avg

    variance = ((x_centered).T @ (x_centered)).sum() / n

    return x_centered / math.sqrt(variance)

# print(normalize_vector(hs_gpa))
x = torch.stack((normalize_vector(hs_gpa), normalize_vector(sat)), dim=1).squeeze(2)

y = normalize_vector(penn_gpa)

# gradient descent 

i = 0
step_size = 0.1
iterations = 100

def train(x, y, iterations, step_size):
    i = 0
    loss_history = []

    parameters = torch.tensor([[1.],[1.]])

    while i < iterations:

        # Record loss
        loss_history.append(root_mean_squared_error(x, y, parameters))

        # calculate gradient
        gradient = x.T @ (y - torch.matmul(x, parameters)) / len(y)

        # calculate new params
        parameters = parameters + step_size * gradient

        i += 1

    return [parameters, loss_history]
    
[parameters, loss_history] = train(x, y, iterations, step_size)

plt.plot(loss_history)
plt.title("Root Mean Squared Error")
plt.xlabel("Iteration")
plt.ylabel("RMSE")

print(f"trained params: {parameters[0].item():.3f}, {parameters[1].item():.3f}")

# Trying different step sizes

step_size = 1

[parameters, loss_history] = train(x, y, iterations, step_size)

plt.plot(loss_history)

step_size = 0.01

[parameters, loss_history] = train(x, y, iterations, step_size)

plt.plot(loss_history)


plt.savefig("root_mean_squared_error.jpg")