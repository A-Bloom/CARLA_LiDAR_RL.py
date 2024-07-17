

def afunc(a=1, b=2, **kwargs):
    print(a+b)


args = {'a': 2, 'c': 7}

afunc(**args)
