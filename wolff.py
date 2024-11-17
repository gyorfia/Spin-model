import queue
import Graphics as gfx
import numpy as np
from numpy import save, load
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    neighboring_sum=sum_of_neigbors(spinMat, i, j, N)
    E1 = J * spinMat[i][j] * (neighboring_sum)
    E2 = J * (-1) * spinMat[i][j] * (neighboring_sum)  # energy if flipped
    return E2-E1

#create a cluster and adding spins based on its state and the Boltzman factor
def create_cluster(spinMat, i,j, t, rng1_fn):
    cluster_fn=[]
    cluster_set=set()

    cluster_fn.append([i, j])
    cluster_set.add((i,j))

    queue=[(i,j)]
    while queue:
            c_i,c_j= queue.pop()


            for ni, nj in neighbors(c_i, c_j, N):
                if (ni,nj) not in cluster_set and spinMat[ni][nj]==spinMat[i][j] :
                    d_E=d_E_count(spinMat, ni,nj,N)

                    factor_fn=np.exp(-d_E/t)
                    if d_E<0 or factor_fn> (rng1_fn.random()):
                        cluster_fn.append([ni,nj])
                        cluster_set.add((ni,nj))
                        queue.append((ni,nj))

    return np.array(sorted(cluster_fn, key=lambda x:(x[0], x[1])))

#the boundary spins of a cluster, not in use
def cluster_boundary(cluster, N):
    cluster_set=set(tuple(spin) for spin in cluster)
    c_boundary=[]

    for spin in cluster:
        c_i,c_j=spin
        for ni, nj in neighbors(c_i, c_j, N):
            if (ni, nj) not in cluster_set:
                c_boundary.append([ni, nj])

    c_boundary=np.array(c_boundary)
    return c_boundary

#The neigboring spins of a cluster not in use
def cluster_neighbors(cluster,N):
    c_neighbors=[]

    for i in range(len(cluster)):
        c_i,c_j=cluster[i]
        cell_neighbors=neighbors(c_i, c_j, N)
        for n in cell_neighbors:
            if not np.any(np.array_equal(n,spin)for spin in cluster):
                c_neighbors.append([c_i, c_j])

    c_neighbors=np.array(c_neighbors)
    return c_neighbors

#flips the cluster, not in use (Wrong)
def flip_cluster(spinMat, cluster, N, rng1_fn, t):
    J=-1
    c_boundary=cluster_boundary(cluster, N)
 
    E1_cluster=0
    E2_cluster=0
    d_E=0

    for spin in c_boundary:
        c_b_i, c_b_j=spin
        neighbors_sum=sum_of_neigbors(spinMat, c_b_i, c_b_j, N)

        E1_cluster+=J*spinMat[c_b_i][c_b_j]*neighbors_sum
        E2_cluster+=J*(-1)*spinMat[c_b_i][c_b_j]*neighbors_sum
    
    d_E=E2_cluster-E1_cluster

    factor_fn=np.exp(-d_E/t)
    
    if d_E<0 or factor_fn>(rng1_fn.random()):
        for spin in cluster:
            c_i,c_j=spin
            spinMat[c_i][c_j]*=-1


    return  spinMat
#flips the entire cluster
def flip_c(spinMat, cluster, N):
    for i, j in cluster:
        spinMat[i][j]*=-1
    return spinMat


# INIT
N = 400
J = -1
k = np.pow(N, 3) # originally: k=np.pow(N,2.179)
# Change figure dimension in constructor: Graphics(nRows, nColumns)
gfx = gfx.Graphics(2, 2)
spinMat = np.random.choice([-1, 1], size=(N, N))
flips = 0
seed = 1234
np.random.seed(seed)
rng1 = np.random.default_rng(seed)
tc = 2 / (np.log(1 + np.sqrt(2)))


# SPINMAT CALCULATIONS
for it in tqdm(range(int(k)), desc="Generating spinMat"):
    i = rng1.integers(0, N)
    j = rng1.integers(0, N)
    
    cluster= create_cluster(spinMat, i, j, tc, rng1)

    spinMat=flip_c(spinMat, cluster, N)

    flips+=1

gfx.Black_White(spinMat, N)

plt.show()  # Show the final plot after all updates
