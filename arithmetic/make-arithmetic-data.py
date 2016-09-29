import random

for i in range(5000):
    a = random.randint(0,1000)
    b = random.randint(0,1000)
    print("{}+{},{}".format(a, b, a+b))