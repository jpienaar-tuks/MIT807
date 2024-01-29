import matplotlib.pyplot as plt
import numpy as np
from ioh import get_problem, ProblemClass

def make3Dplot(f, ax):
    problem=get_problem(f,1,2,ProblemClass.BBOB)
    X=np.arange(-5,5,0.25)
    Y=np.arange(-5,5,0.25)
    X,Y = np.meshgrid(X,Y)
    Z=np.zeros(shape=(40,40))
    for i, y in enumerate(np.arange(-5,5,0.25)):
            for j, x in enumerate(np.arange(-5,5,0.25)):
                    Z[i,j]=problem([x,y])
    surf = ax.plot_surface(X,Y,Z,cmap='viridis')
    ax.set_title(problems[f])

def makeContour(f, ax):
    problem=get_problem(f,1,2,ProblemClass.BBOB)
    X=np.arange(-5,5,0.1)
    Y=np.arange(-5,5,0.1)
    X,Y = np.meshgrid(X,Y)
    Z=np.zeros(shape=X.shape)
    for i, y in enumerate(np.arange(-5,5,0.1)):
            for j, x in enumerate(np.arange(-5,5,0.1)):
                    Z[i,j]=problem([x,y])
    surf = ax.contourf(X,Y,Z,20,cmap='viridis')
    #plt.gcf().colorbar(surf, ax=ax)
    ax.set_title(problems[f])

problems = ProblemClass.BBOB.problems

fig, ax = plt.subplots(4,6,subplot_kw={"projection": "3d"})
for i, ax_ in enumerate(ax.ravel()):
    make3Dplot(i+1, ax_)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(4,6)
for i, ax_ in enumerate(ax.ravel()):
    makeContour(i+1, ax_)
fig.tight_layout()
plt.show()
