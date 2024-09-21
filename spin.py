import Graphics as gfx
import numpy.random as random

# INIT
N = 10
J = 1
gfx = gfx.Graphics(N)
random.seed(1)
spinMat = random.choice([-1, 1], size=(N, N))
flips = 0

# LOGIC
for k in range(N): # how many loops on the whole grid
    for i in range(N):
        for j in range(N):
            E1 = J*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N]) # negative indexing is implemented
            E2 = J*(-1)*spinMat[i][j]*(spinMat[i-1][j] + spinMat[(i+1)%N][j] + spinMat[i][j-1] + spinMat[i][(j+1)%N]) # energy if flipped
            if(E2 < E1):
                spinMat[i][j] *= -1
                flips +=1
    gfx.UpdateModel(spinMat)
    print(f"Updating, flip count: {flips}\n")
gfx.FinalPlot()