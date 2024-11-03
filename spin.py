import Graphics as gfx
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def BoxIsFilled(spinMat, i0, i1, j0, j1):
    '''
    (spinMat, starting row, ending row (incl.), starting column, ending column (incl.))
    '''
    submatrix = spinMat[i0:i1+1, j0:j1+1]
    if np.all(submatrix == 1) or np.all(submatrix == -1):
        return False
    return True

# INIT
N = int(2**8)
J = -1
iteration = 0
k = np.pow(N, 2.8)  # originally: k=np.pow(N,2.179)
# Change figure dimension in constructor: Graphics(nRows, nColumns)
gfx = gfx.Graphics(1, 2)
spinMat = np.random.choice([-1, 1], size=(N, N))
flips = 0
seed = 1234
np.random.seed(seed)
rng1 = np.random.default_rng(seed)
tc = 2 / (np.log(1 + np.sqrt(2)))

# SPINMAT CALCULATIONS
for _ in tqdm(range(int(k)), desc="Generating spinMat"):
    i = rng1.integers(0, N)
    j = rng1.integers(0, N)
    E1 = J * spinMat[i][j] * (spinMat[i - 1][j] + spinMat[(i + 1) % N][j] + spinMat[i][j - 1] + spinMat[i][(j + 1) % N])
    E2 = J * (-1) * spinMat[i][j] * (spinMat[i - 1][j] + spinMat[(i + 1) % N][j] + spinMat[i][j - 1] + spinMat[i][(j + 1) % N])  # energy if flipped
    factor = np.exp((-(E2 - E1)) / tc)
    if E2 - E1 < 0 or factor > rng1.random():
        spinMat[i][j] *= -1
        flips += 1

# DIMENSION COUNTING
eps = 64 # first length of the box
n = N // eps # first size of the box-grid
xs_list = []
ys_list = []
while eps >= 2: # last length of the box
    nBox = 0
    for i in range(n):
        for j in range(n):
            if BoxIsFilled(spinMat, eps * i, eps * (i + 1) - 1 if i != n - 1 else N - 1, eps * j, eps * (j + 1) - 1 if j != n - 1 else N - 1):
                nBox += 1
    ys_list.append(np.log(nBox))
    xs_list.append(np.log(eps))
    print(f"Current box size: {eps}")
    eps //= 2
    n = N // eps

# PLOTTING
xs = np.array(xs_list)
ys = np.array(ys_list)
gfx.Black_White(spinMat, N)
model = LinearRegression().fit(xs.reshape((-1, 1)), ys)
gfx.Plot(xs, model.predict(xs.reshape((-1, 1))), f"Dimension: {abs(model.coef_)}", "log(epsilon)", "log(N(epsilon))", False)
gfx.Scatter_Current(xs, ys)
print(f"\nModel function: f(x)= {model.intercept_:0.2f} + {model.coef_[0]:0.2f}x")
print(f"Coefficient of determination: {model.score(xs.reshape(-1, 1), ys.reshape(-1, 1)):0.4f}")
gfx.Show() # <-> plt.show()