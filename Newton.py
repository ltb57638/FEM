from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.solvers import *

ngsglobals.msg_level = 1

# generate a triangular mesh of mesh-size 0.1
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

# H1-conforming finite element space
fes = H1(mesh, order=1, dirichlet="left|right|bottom|top")

# coefficient function
f = (-2*x*x)*((-1+x)**2)*(y**2)*((-1+y)**2)*(x**2 + y**2 - x - y)

# define trial- and test-functions
u = fes.TrialFunction()
v = fes.TestFunction()
gfu = GridFunction(fes)
g = 0

# the bilinear-form 
a = BilinearForm(fes)
a += SymbolicBFI((u**2*grad(u))*grad(v) - f*v)
#a += (grad(u)*grad(v) - x*v)*dx
a.Assemble()

gfu.Set(x*(1-x)*y*(1-y))
#gfu.Set(g, BND)

Newton(a,gfu,freedofs=gfu.space.FreeDofs(),maxit=1000,maxerr=1e-7,dampfactor=0.5,printing=True)
Draw(gfu,mesh,"u")

exact = x*(1-x)*y*(1-y)
print ("L2-error:", sqrt (Integrate ( (gfu-exact)*(gfu-exact), mesh)))
