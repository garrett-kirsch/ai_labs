import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)
plt.style.use('default')

data = torch.from_numpy(np.genfromtxt('Penn-GPA-and-HS-Data.csv', delimiter=",", skip_header=1, dtype=float))

