
l = [('c', 2), ('b', 2), ('a', 1), ('b', 1), ('c', 1)]

ret = sorted(l, key=lambda x: (x[0], x[1]))
print(ret)
