import matplotlib.pyplot as plt


a = [1,1,2,2]

b = [0.1,0.2,0.1,0.2]

c = [5,2,10,4]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(a, b, c, marker='o')

ax.set_xlabel('env')
ax.set_ylabel('lr')
ax.set_zlabel('reward')

# plt.savefig('Output/lr.jpg')

plt.show()