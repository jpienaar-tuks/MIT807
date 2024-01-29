import numpy as np
rot90Clock = np.array([[0,1],[1,0]])
rot90AntiClock = np.array([[0,-1],[-1,0]])

def hilbertcurve(order):
    if order == 1:
        return np.array([[0,0],[0,1],[1,1],[1,0]])
    else:
        scale = 2**(order-1)-1
        curves = [hilbertcurve(order -1) for i in range(4)]
        curves[0] = ((curves[0] - scale/2)@rot90Clock)+scale/2
        curves[3] = ((curves[3] - scale/2)@rot90AntiClock)+scale/2
        translate = 2**(order-1)
        curves[1] += [0         , translate]
        curves[2] += [translate , translate]
        curves[3] += [translate , 0]
        return np.concatenate(curves)
        
        
