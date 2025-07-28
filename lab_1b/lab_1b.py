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

# Normalize inputs

def normalize_vector(x):
    n = len(x)
    average = (x.sum() / n)
    x_avg = average.expand(n).unsqueeze(1)
    print(x_avg.shape)
    x_centered = x - x_avg
    print(x.shape)
    variance = ((x_centered).T @ (x_centered)).sum() / n

    return x_centered / math.sqrt(variance)

# print(normalize_vector(hs_gpa))