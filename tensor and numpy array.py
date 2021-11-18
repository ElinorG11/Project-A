# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

a = np.random.rand(9,8)
b = np.random.rand(9,8)

print(a)

  
plt.scatter(a, b)
plt.title("example")
plt.xlabel("a")
plt.ylabel("b")
plt.show()

import torch

"""a = np.array(a)
b= np.array(b)"""
#define a PyTorch Tensor usning Python List
c = torch.tensor([a, b])
print(c)
# compute mean, std, and var
m = torch.mean(c)
s = torch.std(c)
v = torch.var(c)
# print mean, std, and var
print("Mean:{}\n std: {}\n Var: {}".format(m,s,v))

