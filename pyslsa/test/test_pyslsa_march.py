import pyslsa

s = pyslsa.build_SCG([(1,2,3), (4,5,6,7)])
s.print()
s.print_D(2)

r = pyslsa.build_SCG([(1,2,3), (1,5,6,8)])
print(pyslsa.KL(r,s, 1, 1))
