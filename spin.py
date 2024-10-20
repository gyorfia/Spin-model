import Graphics as gfx
import numpy as np
import math

def BoxIsFilled(spinMat, i0, i1, j0, j1):
    '''
    (spinMat, starting row, ending row (incl.), starting column, ending column (incl.),)
    '''
    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            if(spinMat[i][j] == -1):
                return True
    return False

#INIT
N = 500
J = -1
iteration=0
k=math.pow(N,2.179)
# Change figure dimension in constructor: Graphics(nRows, nColumns)
gfx = gfx.Graphics(1, 2)
spinMat = np.random.choice([-1, 1], size=(N, N))
flips = 0
seed = 1234
np.random.seed(seed)
rng1=np.random.default_rng(seed)
tc=2/(np.log(1+np.sqrt(2)))
#LOGIC
while(iteration <k): # how many loops on the whole grid
        i=rng1.integers(0,N)
        j=rng1.integers(0,N)
        E1 = J*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N])
        E2 = J*(-1)*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N]) # energy if flipped
        factor=np.exp((-(E2-E1))/tc)
        if(E2-E1 <0 or factor> rng1.random()):
            spinMat[i][j] *= -1
            flips +=1
        iteration +=1
        print(f"Generating spinMat: {(100*iteration/k):.0f}%")
eps = 10 # first length of the box
n = N//eps # first size of the box-grid
xs = []
ys = []
while eps >= 2: # last length of the box
    nBox = 0
    for i in range(n):
        for j in range(n):
            if(BoxIsFilled(spinMat, eps*i, eps*(i+1)-1 if i != n-1 else N-1, eps*j, eps*(j+1)-1 if j != n-1 else N-1)):
                nBox += 1
    ys.append(math.log(nBox))
    xs.append(-math.log(eps))
    print(f"Current box size: {eps}")
    eps -= 1
    n = N//eps if eps!=0 else 0

# Plotting
gfx.Black_White(spinMat, iteration)
lastI = len(xs)-1
dimension = (ys[lastI]-ys[0])/(xs[lastI]-xs[0]) # slope
gfx.Plot(xs, ys, f"Dimension: {dimension}", "-log(epsilon)", "log(N(epsilon))", True)
gfx.Show() # <-> plt.show()
