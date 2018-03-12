import pycuslsa

s = pycuslsa.build_SCG([(1,2,3), (4,5,6,7)])
s.print()
s.print_D(2)


r = pycuslsa.build_SCG([(1,2,3), (1,5,6,8)])
print(pycuslsa.KL(r,s, 1, 1))
print(s.L_dim(1))