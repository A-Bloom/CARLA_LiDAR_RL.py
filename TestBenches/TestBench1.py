from gymnasium import spaces
import numpy as np

observation_space = spaces.Dict({'lidar': spaces.Box(high=1, low=-1, shape=(2, 2)), 'other': spaces.Box(high=np.inf, low=-np.inf, shape=(2,))})


print(np.zeros(2))


class AClass:

    def __init__(self, var1, var2, listing):
        self.var1 = var1
        self.var2 = var2
        self.listing = listing

    def do_stuff(self):

        var3 = 132

        for thing in self.listing:
            if thing in locals():
                print(locals()[thing])
            elif thing in dir(self):
                print(getattr(self, thing))
            else:
                print("This variable does not exist")
