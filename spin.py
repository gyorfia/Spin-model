import Graphics as gfx
import numpy as np

#INIT
N = 10
TPrecision = 100
J = -1
gfx = gfx.Graphics(N)
spinMat = np.random.choice([-1, 1], size=(N, N))
flips = 0
seed = 120
np.random.seed(seed)
rng1=np.random.default_rng(seed)
tx = np.linspace(0.1, 5, TPrecision)
ty = np.zeros(TPrecision)

#LOGIC
for l in range(len(tx)):
    for k in range(N**4): # how many loops on the whole grid
        i=rng1.integers(0,N)
        j=rng1.integers(0,N)
        E1 = J*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N])
        E2 = J*(-1)*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N]) # energy if flipped
        factor=np.exp((-(E2-E1))/tx[l])
        if(E2-E1 <0 or factor> rng1.random()):
            spinMat[i][j] *= -1
            flips +=1
    # ???
    for i in range(N):
        for j in range(N):
            ty[l] += J*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N])
    print(f"Progress: {l*100/TPrecision:.0f}%\n")
ty = ty / (N*N)
gfx.Plot(tx, ty)
