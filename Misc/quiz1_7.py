import numpy as np
# b = 2.005
# num = 25
b = float(input('b='))
num = int(input('how many elements are there?'))

# summing everything without A
pa = np.empty(num)
for i in range(num):
	pa[i] = b**i

A = 1/np.sum(pa)
p = pa*A #probability for every element

sum = 0
for index in range(num):
	if (index%2==1):
		sum = sum + p[index]

print('The answer is',sum)