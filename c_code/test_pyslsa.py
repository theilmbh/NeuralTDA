import pyslsa

t = pyslsa.Simplex()
v = pyslsa.Simplex()
for i in range(4):
    t.add_vertex(i)

for i in range(5, 20):
    v.add_vertex(i)

s = pyslsa.SCG()
r = pyslsa.SCG()
s.add_max_simplex(t)
r.add_max_simplex(v)
r.add_max_simplex(t)    

s.print()
r.print()

s.print_L(1);
r.print_L(1);

print(pyslsa.KL(s,r, 1, -0.15))
