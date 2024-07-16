stuff = {'a':1,
         'b':2}
more_stuff = {'c': 3, 'd': 100}


def afunc(a=2, b=3, c=4, d=5, e=6):
    print(f"{a+b+c+d+e}")


for i in stuff:
    locals()[i] = stuff[i]

print(b)
