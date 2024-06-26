import matplotlib.pyplot as plt


a = [2,13,4]

b = [2,5,2]

plt.plot(a,b, 'o')

plt.xlabel('lr')
plt.ylabel('reward')

plt.savefig('Output/lr.jpg')

plt.show()
