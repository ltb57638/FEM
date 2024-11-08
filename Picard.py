from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.solvers import *
from netgen.geom2d import SplineGeometry

ngsglobals.msg_level = 1
# Generate the background mesh
square = SplineGeometry()
square.AddRectangle((2, 2), (3, 3), bcs= ["1","2","3","4"])
mesh = Mesh(square.GenerateMesh(maxh=0.4/4, quad_dominated=False))

# H1-conforming finite element space
fes = H1(mesh, order=1, dirichlet="1|2|3|4")
# coefficient function
exactu = x**2+y**2
exact_g = CoefficientFunction((2*x,2*y))
source = -8*(x**2 + y**2 + 1)*x**2 - 4*(x**2 + y**2 + 1)**2 - 8*(x**2 + y**2 + 1)*y**2

# define trial- and test-functions
u = fes.TrialFunction()
v = fes.TestFunction()

a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

def nonlinearity(u):
    return (u+1.0)*(u+1.0)

gfu_init = GridFunction(fes)
gfu_init.Set(exactu, BND)

gfu_old = GridFunction(fes)
gfu_old.vec.data = gfu_init.vec.data

gfu_new = GridFunction(fes)

#Draw(gfu_old)
maxits = 1

for it in range(maxits):
    print ("Iteration {:3}  ".format(it),end="")
    rhs = LinearForm(fes)
    rhs += (source/nonlinearity(gfu_old))*v*dx
    rhs_assembled = rhs.Assemble()
    
    gfu_new.vec.data = a.mat.Inverse(fes.FreeDofs()) * rhs_assembled.vec
    gfu_old.vec.data = gfu_new.vec.data
    print ("L2-error:", sqrt (Integrate ( (gfu_new-exactu)*(gfu_new-exactu), mesh)))
    print ("H1-error:", sqrt (Integrate ( (grad(gfu_new)-exact_g)**2, mesh)))

Draw(gfu_new)

#print ("L2-error:", sqrt (Integrate ( (gfu_new-exactu)*(gfu_new-exactu), mesh)))
#print ("H1-error:", sqrt (Integrate ( (grad(gfu_new)-exact_g)**2, mesh)))

