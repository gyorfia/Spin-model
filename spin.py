from cProfile import label
import queue
import Graphics as gfx
import numpy as np
from numpy import save, load
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

#The indices os neighboring spins
def neighbors(i,j,N):
    cell_neighbors=np.array([
        [i-1, j], 
        [(i+1)%N, j],
        [i, j-1],
        [i, (j+1)%N]
    ])
    return cell_neighbors

#The sum of the neighboring spins
def sum_of_neigbors(spinMat, i,j,N):
    cell_neighbors=neighbors(i,j,N)
    sum=np.sum(spinMat[cell_neighbors[:,0], cell_neighbors[:,1]])
    return sum

#calculate the energy difference before and after flips 
def d_E_count(spinMat, i, j, N, J=-1):
    neighboring_sum=sum_of_neigbors(spinMat,i,j,N)
    E1 = J * spinMat[i][j] * (neighboring_sum)
    E2 = J * (-1) * spinMat[i][j] * (neighboring_sum)  # energy if flipped
    return E2-E1

#create a cluster and adding spins based on its state and the Boltzman factor
def create_cluster(spinMat, i,j, t, rng1_fn, N):
    cluster_set=set([(i,j)])

    queue=deque([(i,j)])

    spin_val=spinMat[i][j]
    energy_dif=0
    while queue:
            c_i,c_j= queue.popleft()


            for direction in range(4):
                ni, nj=neighbors(c_i,c_j,N)[direction]
                if(ni, nj) in cluster_set:
                    continue
                if spinMat[ni][nj]==spin_val :
                    d_E=d_E_count(spinMat, ni,nj,N,)

                    factor_fn=np.exp(-d_E/t)
                    if d_E<0 or factor_fn> (rng1_fn.random()):
                        cluster_set.add((ni,nj))
                        queue.append((ni,nj))
                        energy_dif+=d_E
    
    if len(cluster_set) < 3:
        return None, 0

    cluster=np.array(list(cluster_set))

    return cluster, energy_dif

#grow an allready created cluster
def grow_cluster(spinMat, cluster, N, t, rng1_fn):
    energy_dif=0
    new_spins=[]
    cluster_set = set(tuple(spin) for spin in cluster)

    for (i, j) in cluster:
         for direction in range(4):
            ni, nj = neighbors(i, j, N)[direction]
            if (ni, nj) not in cluster:
                d_E=d_E_count(spinMat, ni, nj, N)
                factor_fn=np.exp(-d_E/t)
                if d_E< 0 or factor_fn> rng1_fn.random():
                    new_spins.append((ni, nj))
                    energy_dif+= d_E
    
    cluster_set.update(new_spins)
    
    cluster = np.array(list(cluster_set))

    return cluster, energy_dif

#flips the cluster
def flip_cluster(spinMat, cluster, N, rng1_fn, t, energy_dif):
    J=-1
    
    
    if energy_dif<0 or np.exp(-energy_dif/t)>(rng1_fn.random()):
        for (i, j) in cluster:
            spinMat[i][j]*=-1

        energy_dif*=-1

    return  spinMat, energy_dif

def Wolff(spinMat, N, k, t, rng1_fn, flips=0):
    energy_dif=0
    cluster_list=[]
    visited_clusters = set()

    for it in tqdm(range(int(k)), desc="Wolff"):
        i = rng1_fn.integers(0, N)
        j = rng1_fn.integers(0, N)
    
        cluster, energy_dif = create_cluster(spinMat, i, j, t, rng1_fn, N)
        if cluster is None:
            continue
        else:
            spinMat, energy_dif = flip_cluster(spinMat, cluster, N, rng1_fn, t, energy_dif)
            flips += 1

    return spinMat, flips

def Metropolis(spinMat, N, k, t, rng1_fn, flips=0):

    for it in tqdm(range(int(k)), desc="Metropolis"):
        i=rng1_fn.integers(0,N)
        j=rng1_fn.integers(0,N)

        d_e=d_E_count(spinMat, i,j, N, J=-1)
        factor=np.exp(-d_e/t)

        if(d_e<0 or factor>(rng1_fn.random())):
            spinMat[i][j]*=-1
            flips+=1

    return spinMat, flips

def BoxIsFilled(spinMat, i0, i1, j0, j1):
    '''
    (spinMat, starting row, ending row (incl.), starting column, ending column (incl.))
    '''
    submatrix = spinMat[i0:i1+1, j0:j1+1]
    sum_submatrix=np.sum(submatrix)
    #print(f"Sum of absolute values: {sum_submatrix}, Expected: {(i1 - i0 + 1) * (j1 - j0 + 1)}") 
    if sum_submatrix>=(i1-i0+1)*(j1-j0+1) or sum_submatrix<=-(i1-i0+1)*(j1-j0+1)  :
        return False
    return True

def CountDimension(spinMat):
    N=spinMat.shape[0]
    eps = 10 # first length of the box
    #n = N // eps # first size of the box-grid
    xs_list = []
    ys_list = []
    while eps >= 2: # last length of the box
        nBox = 0
        for j in range(N-eps+1):
            for i in range(N-eps+1):
                if(BoxIsFilled(spinMat, i, eps+i-1, j, eps+j-1)):
                    nBox+=1
        if nBox>0:
            ys_list.append(np.log(nBox))
            xs_list.append(np.log(eps))
        eps -=1
        #n = N // eps
        
    xs = np.array(xs_list)
    ys = np.array(ys_list)

    if(np.any(np.isnan(ys)) or np.any(np.isinf(ys))):
        return None
     
    model=LinearRegression().fit(xs.reshape((-1, 1)), ys)
    fractal_dimension=abs(model.coef_[0])
    return fractal_dimension,xs, ys, model

def magnetization(spinMat):
    return np.sum(spinMat)/np.size(spinMat)

# INIT
N = 10
J = -1
iteration = 0
k = 10* int(np.pow(N, 4))  # originally: k=np.pow(N,2.179)
# Change figure dimension in constructor: Graphics(nRows, nColumns)
gfx = gfx.Graphics(2, 2)
spinMat = np.random.choice([-1, 1], size=(N, N))
spinMat_metropolis=np.copy(spinMat)
spinMat_wolff=np.copy(spinMat)

flips = 0
flips_metropolis=flips
flips_wolff=flips

M_metropolis=0
magnetization_metropolis=[]
M_wolff=0
magnetization_wolff=[]

t=np.linspace(1.5,3.5,50)

seed = 1234
np.random.seed(seed)
rng1 = np.random.default_rng(seed)
tc = 2 / (np.log(1 + np.sqrt(2)))
t=0.5

# SPINMAT CALCULATIONS t:

spinMat_wolff, flips_wolff=Wolff(spinMat,N, k, t, rng1, flips=0)
gfx.Black_White(spinMat_wolff, N)

plt.show()

'''
for t_v in t:
    spinMat_metropolis, flips_metropolis= Metropolis(spinMat,N,k, t_v, rng1, flips=0)
    magnetization_metropolis.append(magnetization(spinMat_metropolis))
    print (magnetization(spinMat_metropolis))
gfx.Black_White(spinMat_metropolis, N)
plt.plot(t, magnetization_metropolis, label="Metropolis")
plt.xlabel("t [\u00B0C]")
plt.ylabel("M [A/m^2]")
plt.legend()

for t_v in t:
    spinMat_wolff, flips_wolff=Wolff(spinMat,N, k, t_v, rng1, flips=0)
    magnetization_wolff.append(spinMat_wolff)
    print (magnetization(spinMat_wolff))
gfx.Black_White(spinMat_wolff, N)

plt.plot(t, magnetization_wolff, label="Wolff")
plt.xlabel("t [\u00B0C]")
plt.ylabel("M [A/m^2]")
plt.legend()

plt.show()

loadMat=load("test_fractal.npy")
xs=np.array
ys=np.array

dimension, xs,ys, model=CountDimension(loadMat)

# PLOTTING
dimension_str = f"Dimension: {abs(model.coef_[0]):.10f}"
gfx.Black_White(loadMat, N)
gfx.Plot(xs, model.predict(xs.reshape((-1, 1))), dimension_str, "log(epsilon)", "log(N(epsilon))", False)
gfx.Scatter_Current(xs, ys)
print(f"\nModel function: f(x)= {model.intercept_:0.2f} + {model.coef_[0]:0.2f}x")
print(f"Coefficient of determination: {model.score(xs.reshape(-1, 1), ys.reshape(-1, 1)):0.4f}")
gfx.Plot(xs, ys, f"Dimension(iteration)", "logN(iteration)", "dimension", False)
gfx.Show() # <-> plt.show()
'''