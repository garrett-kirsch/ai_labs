import numpy as np
import matplotlib.pyplot as plt
import torch
import math

data = data = torch.from_numpy(
           np.genfromtxt (
               'Penn-GPA-and-HS-Data.csv',
               delimiter = ",",
               skip_header=1,
               dtype = float ) )


hs_gpa      = data[:,1]
sat_scores  = data[:,2]
penn_gpa    = data[:,4]

plt.plot(hs_gpa, penn_gpa, ".")
plt.title("Penn GPA vs. HS GPA")
plt.xlabel("HS GPA")
plt.ylabel("Penn GPA")
plt.savefig("penn_gpa_v_hs_gpa.jpg")

plt.clf()

plt.plot(sat_scores, penn_gpa, ".")
plt.title("Penn GPA vs. SAT Scores")
plt.xlabel("SAT Scores")
plt.ylabel("Penn GPA")
plt.xlim([1380, 1620])
plt.savefig("penn_gpa_v_sat_scores.jpg")

# find a* = sum (x_i * y_i) / sum (x_i^2)

num = 0
denom = 0
i = 0

# iterative approach
# while i < len(penn_gpa):
#     num += penn_gpa[i] * hs_gpa[i]
#     denom += hs_gpa[i] * hs_gpa[i]
#     i += 1

# alpha_star = num / denom

# matrix multiplication approach

# x = hs_gpa.unsqueeze(1)
# y = penn_gpa.unsqueeze(1)
# print(f"hs_gpa shape: {hs_gpa.shape:.3f}")
# print(f"x shape: {x.shape:.3f}")
# print(f"penn_gpa shape: {penn_gpa.shape:.3f}")
# print(f"y shape: {y.shape:.3f}")

x = hs_gpa.unsqueeze(1)
y = penn_gpa.unsqueeze(1)

alpha_star = (x.T @ y) / (x.T @ x)


print(f"alpha_star = {alpha_star.item():.3f}")
print(f"Predicted_Penn_GPA = {alpha_star.item():.3f} * High_School_GPA ")

y_hat = x * alpha_star

plt.clf()

plt.plot(x, y_hat)
plt.plot(hs_gpa, penn_gpa, ".")
plt.title("Penn GPA vs. HS GPA with Trendline1")
plt.xlabel("HS GPA")
plt.ylabel("Penn GPA")
plt.savefig("penn_gpa_v_hs_gpa_with_trendline.jpg")

v = y - alpha_star * x

root_mean_squared_error = math.sqrt((v.T @ v) / len(x))

print(f"root_mean_squared_error = {root_mean_squared_error:.3f}")

# Linear Regression

x = data[:,1:3]

print(x)

w_star = torch.linalg.inv(x.T @ x) @ x.T @ y

print(w_star)

# w_start = (x.T @ y) / torch.linalg.det(x.T @ x)

# print(w_star)