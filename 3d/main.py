import time
import numpy as np 
from matplotlib import pyplot as plt
from scipy.linalg import norm

neuron = [] 

def interpolate_to_parent(start, end, linspace_count):
    """ 
        Draw line between point 1 and point 2
        Assume start end numpy vectors 
        based on neurom
    """

    v = end - start
    length = norm(v)
    v = v / length # Make v a unit vector

    l = np.linspace(0, length, linspace_count) 

    return np.array([start[i] + v[i] * l for i in range(3)])


def scale_image(pixels, size):
    """
        Scale points to [0..size]

        points = np arr
    """
    x_min, x_max = np.amin(pixels[:,0]), np.amax(pixels[:,0])
    y_min, y_max = np.amin(pixels[:,1]), np.amax(pixels[:,1])
    z_min, z_max = np.amin(pixels[:,2]), np.amax(pixels[:,2])
    
    pixels[:,0] -= x_min    
    pixels[:,1] -= y_min
    pixels[:,2] -= z_min
    
    x_max -= x_min
    y_max -= y_min
    z_max -= z_min
    
    scale_factor = size / max(x_max, y_max, z_max) 
    # All points are now between [0..max]

    pixels *= scale_factor
    return pixels

    

start = time.time()

with open('../swc_files/10075.CNG.swc') as f:
    for lines in f:
        if lines[0] == '#':
            continue
        if lines.isspace():
            continue
        neuron.append(list(map(lambda x: float(x), lines.split()))) 

        # LINE ON FORM:
        # Iter TYPE X Y Z RADI PARENT

neuron = np.array(neuron)

pixels = [] 

for n in neuron:
    if n[6] == -1:
        pixels.append(n[2:5])
        continue

    p1 = n[2:5]
    p2 = neuron[int(n[6])-1][2:5]

    pixels.append(p1.tolist())

    X, Y, Z = interpolate_to_parent(p1, p2, 100)

    for i in range(len(X)):
        pixels.append([X[i], Y[i], Z[i]])


pixels = np.array(pixels)

size = 128

pixels = scale_image(pixels, size-1)

pixels = pixels.astype(int)

image = np.zeros((size, size, size), dtype=bool)

for n in pixels:
    x = n[0]
    y = n[1]
    z = n[2]
    image[z,x,y] = 1

figure = plt.figure()
ax = figure.add_subplot(111, projection ='3d') 

z, x, y = image.nonzero()

end = time.time()

print(f'THIS THING TOOK %f TIME' %(end-start))

ax.scatter(x, y, z, s=1)


plt.show()
