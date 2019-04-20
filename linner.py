"""
from scipy import optimize as op
import numpy as np
import math

c=np.array([2,3,-5])
A_ub=np.array([[-2,5,-1],[1,3,1]])
B_ub=np.array([-10,12])
A_eq=np.array([[1,1,1]])
B_eq=np.array([7])
x1=(0,0)
x2=(0,0)
x3=(0,0)
res=op.linprog(-c,A_ub,B_ub,A_eq,B_eq,bounds=(x1,x2,x3))
a = res.x

if type(a) == np.ndarray:
    print("hello")

b = []
print(res)
print(type(a))
print(a)


import re
a='0.0 0.0 0.30626680357797503 0.6937331964220368 0.0 0.0 0.0 0.0 0.0 0.0'
b=re.findall(r'\d+.\d+',a)
sum=0
print(b)
for b1 in b:
    print(b1)
    sum+=float(b1)
print(sum)
"""

import numpy as np

a = np.random.uniform(10,20,(1,))
b = np.random.uniform(10,20,(1,))
c = np.hstack((a,b))

aa = c[0]
print(aa)
