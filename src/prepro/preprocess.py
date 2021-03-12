import numpy as np 
from matplotlib import pyplot as plt
from scipy.linalg import norm
import os

SIZE = 128
UPLOAD_DIR = './src/data'
SWC_FILES_PATH = './swc_files'

class Preprocess():
    
    def __init__(self, path):
        self.path = path
        self.neuron = []

    def interpolate_to_parent(self, start, end, linspace_count):
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


    def interpolate_to_parent_with_radius(self, start, end, r_start, r_end, linspace_count):
        v = end - start
        length = norm(v)
        v = v / length # Make v a unit vector
        res = self.interpolate_to_parent(start, end, linspace_count)

        if r_start > 1:
            o1 = np.random.randn(3)  # normalized orthogonal vector 1
            o1 -= o1.dot(v) * v
            o1 /= np.linalg.norm(o1)
            o2 = np.cross(v, o1)  # normalized orthogonal vector 2

            for i in range(int(r_start)):
                for j in range(int(r_start)):
                    if abs(i) + abs(j) <= int(r_start):
                        line = self.interpolate_to_parent(start + (o1*i + o2*j),
                                                          end + (o1*i + o2*j), linspace_count)
                        res = np.concatenate((res, line), axis=1)

        return res 


    def scale_image(self, pixels, size):
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


    def get_image(self):
        with open(self.path) as f:
            for lines in f:
                if lines[0] == '#':
                    continue
                if lines.isspace():
                    continue
                self.neuron.append(list(map(lambda x: float(x), lines.split()))) 

        neuron = np.array(self.neuron)
        pixels = [] 

        for n in neuron:
            if n[6] == -1:
                pixels.append(n[2:5])
                continue
            p1 = n[2:5]
            r1 = n[5]
            p2 = neuron[int(n[6])-1][2:5]
            r2 = neuron[int(n[6])-1][5]
            pixels.append(p1.tolist())
            X, Y, Z = self.interpolate_to_parent_with_radius(p1, p2, r1, r2, 100)
            for i in range(len(X)):
                pixels.append([X[i], Y[i], Z[i]])

        pixels = np.array(pixels)
        size = SIZE
        pixels = self.scale_image(pixels, size-1)
        pixels = pixels.astype(int)
        image = np.zeros((size, size, size), dtype=bool)

        for n in pixels:
            x = n[0]
            y = n[1]
            z = n[2]
            image[z,x,y] = 1

        return image


    def plot(self, image):
        figure = plt.figure()
        ax = figure.add_subplot(111, projection ='3d') 
        z, x, y = image.nonzero()
        ax.scatter(x, y, z, s=1)
        plt.show()
        

    def save(self, image, filename):
        with open(os.path.join(UPLOAD_DIR, filename[:7] + 'npy'), 'wb') as f:
            np.save(f, image)



def save_all_files():
    for root, dirs, files in os.walk(SWC_FILES_PATH):
        for file in files:
            try:
                p = Preprocess(os.path.join(root, file))
                image = p.get_image()
                p.save(image, file)
                print('DONE:', file)
            except Exception as e:
                print('ERROR:', file, e)


if __name__ == '__main__':
    p = Preprocess('./swc_files/136009.CNG.swc')
    p.plot(p.get_image())
