import Graphics as gfx
import numpy as np
import math

#INIT
N = 500
J = -1
iteration=0
k=math.pow(N,2.179)
gfx = gfx.Graphics()
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
        print(iteration)
gfx.Black_White(spinMat, iteration)
