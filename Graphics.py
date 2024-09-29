import numpy as np
import matplotlib.pyplot as plt

class Graphics:
    def __init__(self, N):
        """
        N by N grid
        """
        '''# Create grid coordinates
        self.X, self.Y = np.meshgrid(np.arange(N), np.arange(N))
        # Define the arrow directions: (0, +1) for spin up, (0, -1) for spin down
        self.U = np.zeros((N, N))  # X-component of arrow is 0 (no horizontal movement)
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        # quiver('grid x coordinates', 'grid y coordinates', 'vector x coordinates', 'vector y coordinates', ...)
        self.quiver = plt.quiver(self.X, self.Y, self.U, self.X, pivot='middle', scale=20, color='black')
        self.ax.set_title('Ising Model Spin Configuration')
        self.ax.invert_yaxis()  # Invert y-axis to match typical matrix visualization
        self.ax.set_aspect('equal')  # Ensure square grid
        self.ax.set_xticks([]) # Remove x-ticks for cleaner look
        self.ax.set_yticks([])  # Remove y-ticks for cleaner look
        plt.ion  # Turn on interactive mode

    def UpdateModel(self, spinMat):
        self.quiver.set_UVC(self.U, spinMat) # Update the quiver with new U and V components
        plt.draw()
        plt.pause(0.0001)

    def FinalPlot(self, spinMat):
        print("Final plot\n")
        self.quiver.set_UVC(self.U, spinMat)
        plt.ioff()
        plt.show()'''

    def Plot(self, xs, ys):
        plt.plot(xs, ys, color="blue")
        plt.show()