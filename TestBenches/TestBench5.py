import json
import zipfile
import os

a_stuff = {'a': 1, 'b': 'two', 'c': 3}
b_stuff = {'d': [4, 5, 6], 'e': 7}
c_stuff = {'f': 8, 'g': ['nine', 'ten']}
d_stuff = None

file = open('testfile.json', 'w')
json.dump([a_stuff, b_stuff, c_stuff, d_stuff], file)
file.close()

archive = zipfile.ZipFile('testzip.zip', 'a')
archive.write('testfile.json', os.path.basename('testfile.json'))
archive.close()

os.remove("testfile.json")


del a_stuff
del b_stuff
del c_stuff
del d_stuff


archive = zipfile.ZipFile('testzip.zip', 'r')
file = archive.open('testfile.json')

a_stuff, b_stuff, c_stuff, d_stuff = json.load(file)

file.close()
archive.close()


print(a_stuff)
print(b_stuff)
print(c_stuff)
print(d_stuff)