class Father():
    def __init__(self):
        self.attribute1 = 0
        self.attribute2 = 0


class Son(Father):
    def addA(self):
        self.attribute1 = 1

f= Father()
son = Son()
son.addA()
print(f.attribute1)
