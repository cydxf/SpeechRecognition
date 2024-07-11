class Test:
    name = " "

    def __init__(self, name):
        self.name = name

    def test(self):
        print(self.name)


# 建立一个新对象，初始化为自己名字，完成修改并提交

p1 = Test("黄嘉希")
p2 = Test("石垒")
p1.test()
p2.test()
