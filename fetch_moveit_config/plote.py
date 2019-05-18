# import matplotlib.pyplot as plt
# import numpy as np
# file = open("/home/ljt/data1", "r")
# lines = file.readlines()
# acc = []
# for i in lines:
#     if i == "\n":
#         continue
#     if i[0] == "t":
#         if float(i.split(",")[2].split("]")[0]) != 0.7500:
#             print i[:-1]
#             # acc.append(float(i[:-1].split(":")[3]))
#             x = float(i[:-1].split(",")[0].split("[")[1])
#             y = float(i[:-1].split(",")[1])
#             z = float(i[:-1].split(",")[2].split("]")[0])
#             acc.append(x)
#     if i[0] == "E":
#         a = float(i[:-1].split(":")[3])
#         l = float(i[:-1].split("|")[1].split(":")[1])
#         # if l < 1:
#         #     acc.append(a)
#
# plt.plot(np.linspace(0, len(acc), len(acc)), acc)
# plt.show()

from fetch_moveit_config.dep_to_position import *

pos, x, y, z = read_data(size=5)
# y = sorted(x)
# plt.plot(np.linspace(0, len(y), len(y)), y)
# plt.show()
print pos[4]
rgb, dep = read_img_dep(pos[3][:-4])
plt.imshow(rgb)
plt.show()
print np.mean(np.argwhere(rgb[:, :, 1] == 0), axis=0)
print rgb[84][76]
print rgb[75][50]
print rgb[200][200]
print rgb[10][10]

