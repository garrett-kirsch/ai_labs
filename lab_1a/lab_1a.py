import numpy as np
import matplotlib.pyplot as plt
import torch

# We can also give nicknames to sublibraries. E.g., to have access to neural network
# functions we re-import the "torch.nn" library and give it the nickname "nn"
import torch.nn as nn

# This is a technicality. It has to be specified.
torch.set_default_dtype(torch.float64)

# Parameters for plots. It controls their appearance. That's all.
plt.style.use('default')
plt.rcParams['font.size'] = '14'


penn_gpa = []
hs_gpa = []
sat = []

data = torch.from_numpy(
           np.genfromtxt (
               'Penn-GPA-and-HS-Data.csv',
               delimiter = ",",
               skip_header=1,
               dtype = float ) )

print(f"\nNumber of students: {data.shape[0]}")
print(f"Number of variables: {data.shape[1]}\n")


penn_gpa = data[:,4]
hs_gpa = data[:,1]
sat = data[:,2]

plt.plot(hs_gpa, penn_gpa)
plt.title("Penn GPA vs HS GPA")
plt.xlabel("HS GPA")
plt.ylabel("Penn GPA")

plt.savefig("penn_gpa_v_hs_gpa.png")

plt.show()

plt.plot(sat, penn_gpa)
plt.title("Penn GPA vs SAT Score")
plt.xlabel("SAT Score")
plt.ylabel("Penn GPA")

plt.savefig("penn_gpa_v_sat.png")

plt.show()