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
# Bendeguz: use much more iterations!
# there are N*N sites --> you need to loop over all sites many times, or at least a few times
for k in range(N**3): # how many loops on the whole grid
    #Bendeguz: instead of going through the whole system pick a random site!!
    #change the upcoming two for loops to randomly picking i and j
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