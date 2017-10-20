import pyslsa

t = pyslsa.Simplex()
for i in range(20):
    t.add_vertex(i)

s = pyslsa.SCG()
s.add_max_simplex(t)
