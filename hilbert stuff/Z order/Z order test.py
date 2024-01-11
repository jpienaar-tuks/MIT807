class Zorder():
    def __init__(self, p, d):
        self.p = p
        self.d = d

    def point2distance(self, point):
        #assert min(point) >= 0
        #assert max(point) <= int('1'*self.p,2)
        #assert all(list(map(lambda x: isinstance(x,int), point)))
        p = [f'{i:b}'.zfill(self.p) for i in point]
        res=''
        for i in zip(*p):
            res+=''.join(i)
        return int(res,2)
        

    def distance2point(self, distance):
        #assert distance <= 2**(self.p*self.d)
        #assert distance >= 0
        #assert isinstance(distance, int)
        binary = f'{distance:b}'.zfill(self.p*self.d)
        dims=[binary[i::self.d] for i in range(self.d)]
        dims=[int(i,2) for i in dims]
        return dims

if __name__== '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2,3)
    idx = ax.ravel()

    for i in range(6):
        Z=Zorder(i+1,2)
        p=[Z.distance2point(j) for j in range(2**(2*(1+i)))]
        p=np.array(p)
        idx[i].plot(p[:,0],p[:,1])

    fig.tight_layout()
    fig.savefig('Z-order curve iterations.png',dpi=200)
    fig.show()
