import numpy as np
import matplotlib.pyplot as plt

class Graphics:
   
    def __init__(self, rows, columns):
        '''
        rows x columns grid of axes
        '''
        # plt.ion() # real time updates
        self.figRows = rows
        self.figColumns = columns
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=columns, squeeze=False, figsize=(10,5))
        for i in range(rows):
            for j in range(columns):
                self.ax[i][j].axis("off")
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        self.nAxes = 0

    def __GetAx(self):
        '''
        Return current ax, intended for private use
        '''
        return self.ax[self.nAxes//self.figColumns][self.nAxes % self.figColumns]

    def CreateArrows(self, N):
        '''
        N
        Only one per Graphics object
        '''
        ax = self.__GetAx()
        ax.axis('on')
        # N by N grid
        # Create grid coordinates
        self.X, self.Y = np.meshgrid(np.arange(N), np.arange(N))
        # Define the arrow directions: (0, +1) for spin up, (0, -1) for spin down
        self.U = np.zeros((N, N))  # X-component of arrow is 0 (no horizontal movement)
        # quiver('grid x coordinates', 'grid y coordinates', 'vector x coordinates', 'vector y coordinates', ...)
        self.quiver = ax.quiver(self.X, self.Y, self.U, self.X, pivot='middle', scale=20, color='black')
        ax.set_title('Ising Model Spin Configuration')
        ax.invert_yaxis()  # Invert y-axis to match typical matrix visualization
        ax.set_aspect('equal')  # Ensure square grid
        ax.set_xticks([]) # Remove x-ticks for cleaner look
        ax.set_yticks([])  # Remove y-ticks for cleaner look
        self.nAxes += 1

    def UpdateArrows(self, spinMat):
        '''
        spinMat
        Update quiver arrows
        '''
        self.quiver.set_UVC(self.U, spinMat) # Update the quiver with new U and V components
        plt.draw()

    def Black_White(self, spinMat, iteration):
        ax = self.__GetAx()
        ax.axis('on')
        ax.imshow(spinMat, cmap='gray')
        ax.set_title(f"Iteration: {iteration}")
        plt.draw()
        self.nAxes += 1
        
    def Plot(self, xs, ys, title="Plot", xaxis="x" ,yaxis="y"):
        '''
        xs, ys, title="Plot", xaxis="x" ,yaxis="y"
        '''
        ax = self.__GetAx()
        ax.set_aspect('auto')
        ax.plot(xs, ys, color='blue')
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.axis('on')
        ax.grid(True)
        self.nAxes += 1
        plt.draw()

    def Show(self):
        # plt.ioff()
        plt.show()