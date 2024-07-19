# This TestBench is a simplified example of how the variableUnion function works.

import copy

a_stuff = {'a': 1, 'b': 'two', 'c': 3}
b_stuff = {'d': [4, 5, 6], 'e': 7}
c_stuff = {'f': 8, 'g': ['nine', 'ten']}
d_stuff = None


def variableUnion(*args, library=[]):
    interim = []

    for variable in args:
        if type(variable) is dict:
            for key, value in variable.items():
                if type(value) is list:
                    if len(library) == 0:
                        for index in value:
                            interim.append({key: index})
                    else:
                        for index in value:
                            for i in range(len(library)):
                                library[i].update({key: index})
                            interim.extend(copy.deepcopy(library))

                    library = interim
                    interim = []
                else:
                    offset = 0
                    if len(library) == 0:
                        library.append({key: value})
                        offset = 1
                    for final in range(offset, len(library)):
                        library[final].update({key: value})

    return library


mid_stuff = variableUnion(b_stuff, d_stuff)
final_stuff = variableUnion(c_stuff, library=mid_stuff)
print(final_stuff)
