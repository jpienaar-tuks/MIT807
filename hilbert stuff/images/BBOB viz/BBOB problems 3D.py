import matplotlib.pyplot as plt
import numpy as np
from ioh import get_problem, ProblemClass

def make3Dplot(f, ax):
    problem=get_problem(f,1,2,ProblemClass.BBOB)
    X=np.arange(-5,5,0.1)
    Y=np.arange(-5,5,0.1)
    X,Y = np.meshgrid(X,Y)
    Z=np.zeros(shape=X.shape)
    for i, y in enumerate(np.arange(-5,5,0.1)):
            for j, x in enumerate(np.arange(-5,5,0.1)):
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
    surf = ax.contour(X,Y,Z,20,cmap='viridis')
    #plt.gcf().colorbar(surf, ax=ax)
    ax.set_title(problems[f])

def makeAll3DPlots():
    for problem in problems.keys():
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        make3Dplot(problem, ax)
        fig.tight_layout()
        fig.savefig(f'{problems[problem]} Surf.png', dpi=400)
        plt.close()

def makeAllContours():
    for problem in problems.keys():
        fig, ax = plt.subplots()
        makeContour(problem, ax)
        fig.tight_layout()
        fig.savefig(f'{problems[problem]} Contour.png', dpi=400)
        plt.close()

problems = ProblemClass.BBOB.problems
makeAll3DPlots()
makeAllContours()


