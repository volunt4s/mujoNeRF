import numpy as np

x = np.array([0.0, 1.0, 0.0])
y = np.array([-0.591, 0.0, 0.806])
z = np.cross(x, y)

print(z)