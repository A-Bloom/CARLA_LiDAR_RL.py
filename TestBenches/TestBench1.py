import math

x = [-2, -2, -2, -3, -3, -3, -3, -3, -2, -2, -1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 3, 3, 3, 3]
y = [-4, -3, -2, -2, -1, 0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 1, 0, -1, -1, -2, -3, -4]

x_range = 4
y_range = 1

closest_collision = 100

for i in range(len(x)):
    if (-x_range <= x[i] <= x_range) and (0 <= y[i] <= y_range):
        if math.hypot(x[i], y[i]) < closest_collision:
            closest_collision = math.hypot(x[i], y[i])

print(closest_collision)

