import matplotlib.pyplot as plt
import numpy as np

file_ob = open('table.txt', 'r')
list1 = file_ob.readlines()
file_ob.close()
for i in range(0, len(list1)):
    list1[i] = list1[i].rstrip('\n')

plt.plot(np.linspace(1, len(list1), len(list1)), list1)
plt.xlabel('Episode X')
plt.ylabel('reward')
plt.show()
