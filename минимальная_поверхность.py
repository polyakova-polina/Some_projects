import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Dot:
    def __init__(self, coords):
        self.x = np.array(coords)
        self.my_triengles = []
        self.layer = 0
        self.past = np.array([0, 0, 0])
        self.move = np.array([0, 0, 0])
    def sum_area(self):
        sum = 0
        for triangle in self.my_triengles:
            sum += triangle.area()
        return(sum)

class Triangle:
    def __init__(self, d1, d2, d3):
        self.d1 = d1.x
        self.d2 = d2.x
        self.d3 = d3.x
        self.layerr = self.get_layerr(d1, d2, d3)
        d1.my_triengles.append(self)
        d2.my_triengles.append(self)
        d3.my_triengles.append(self)

    def get_layerr(self, d1, d2, d3):
        if d1.layer == d2.layer:
            return d1.layer
        elif d1.layer == d3.layer:
            return d1.layer
        elif d3.layer == d2.layer:
            return d2.layer
        else:
            return d2.layer

    def area(self):
        AB = self.d2 - self.d1
        AC = self.d3 - self.d1
        area = 0.5 * np.linalg.norm(np.cross(AB, AC))
        return area

class Layer:
    def __init__(self):
        self.points = []
    def add_point(self, point):
        self.points.append(point)
    def chistca(self, LMAX, N):
        ochist = []
        triangles_for_remove = []
        for i in range(0, N - 1, 2):
            z = - 1
            for j in range(1, 2, 1):
                z = z*(-1)
                if (0 < np.linalg.norm(self.points[i % N].x - self.points[(i + z*j) % N].x) < LMAX) and (
                        self.points[(i + z*j) % N] not in ochist)and (
                        self.points[(i) % N] not in ochist):
                    ochist.append(self.points[(i + z*j) % N])
                    #self.points[i % N].x = (self.points[i % N].x + self.points[(i + z*j) % N].x) / 2
                    for t in self.points[(i + z*j) % N].my_triengles:
                        if t not in triangles_for_remove:
                            if np.all(t.d1 == self.points[(i +z*j) % N].x):
                                t.d1 = self.points[i % N].x
                            if np.all(t.d2 == self.points[(i + z*j) % N].x):
                                t.d2 = self.points[i % N].x
                            if np.all(t.d3 == self.points[(i + z*j) % N].x):
                                t.d3 = self.points[i % N].x
                            if t.area() == 0:
                                triangles_for_remove.append(t)
                            else:
                                self.points[i % N].my_triengles.append(t)
                        #self.points[(i) % N].past = (self.points[(i) % N].past + self.points[(i + z*j) % N].past)/2

        for point in ochist:
            self.points.remove(point)
        return (triangles_for_remove)

class Contour:
    def __init__(self, contour_func, N, n, T_max):
        self.contour = contour_func
        self.layers = []
        self.triangles = []
        self.colvo_layers = N
        self.colvo_points = n
        self.T_max = T_max
    def find_points(self):
        layer0 = Layer()
        LMAX = 0

        for i in range(self.colvo_points):
            t = i * (self.T_max / self.colvo_points)
            layer0.add_point(Dot(self.contour(t)))

            if i != 0:
                previous_t = (i - 1) * (self.T_max / self.colvo_points)
                LMAX += np.linalg.norm(self.contour(t) - self.contour(previous_t)) /((self.colvo_points)*2)

        self.layers.append(layer0)
        return LMAX
    def napr_vector(self, selected_point, points):
        center_point = (points[0].x + points[1].x) / 2
        vec1 = (center_point - points[0].x) / np.linalg.norm(center_point - points[0].x)
        vector = (center_point - selected_point.x) / np.linalg.norm(center_point - selected_point.x)
        vector = vector - vec1 * np.dot(vector, vec1)
        vector = vector / np.linalg.norm(vector)
        return vector
    def draw_extended_point(self, selected_point, points, LMAX, key):
        length = LMAX
        vector = self.napr_vector(selected_point, points)
        if key != 0:
            vector = self.napr_vector(Dot(selected_point.past), points)
        extended_point = selected_point.x + vector*length
        return Dot(extended_point)
    def process(self):
        LMAX = self.find_points()
        key = 0
        for _ in range(self.colvo_layers):
            num = len(self.layers[_].points)
            layer_new = Layer()

            for i in range(num):
                selected_point = self.layers[_].points[(i + 1) % num]
                point1 = self.layers[_].points[i]
                point2 = self.layers[_].points[(i + 2) % num]
                extended_point = self.draw_extended_point(selected_point, [self.layers[_].points[i - _], self.layers[_].points[(i + 2 + _) % num]], LMAX, key)
                extended_point.past = selected_point.x
                extended_point.layer = _ + 1
                layer_new.add_point(extended_point)

                if (i % 2 == 0):
                    tr = Triangle(selected_point, point1, extended_point)
                    self.triangles.append(tr)
                    if i != num - 1:
                        tr = Triangle(selected_point, point2, extended_point)
                        self.triangles.append(tr)
                    else:
                        tr = Triangle(selected_point, layer_new.points[0], extended_point)
                        self.triangles.append(tr)

                    if i != 0:
                        tr = Triangle(point1, extended_point, layer_new.points[i - 1])
                        self.triangles.append(tr)
                else:
                    tr = Triangle(selected_point, extended_point, layer_new.points[i - 1])
                    self.triangles.append(tr)

                    if i == num - 1:
                        tr = Triangle(selected_point, layer_new.points[0], extended_point)
                        self.triangles.append(tr)

            tr_for_remove = layer_new.chistca(LMAX*0.6, len(layer_new.points))
            for t in tr_for_remove:
                self.triangles.remove(t)
            print(len(layer_new.points))

            self.layers.append(layer_new)
            key = 1

        layer_last = Layer()
        center = np.array([0, 0, 0])
        for point in layer_new.points:
            center = center + point.x /len(layer_new.points)

        cent = Dot(center)
        layer_last.points.append(cent)
        self.layers.append(layer_last)
        for i in range(len(layer_new.points) - 1):
            self.triangles.append(Triangle(layer_new.points[i], layer_new.points[i + 1], cent))
        self.triangles.append(Triangle(layer_new.points[0], layer_new.points[len(layer_new.points) - 1], cent))

        for t in self.triangles:
            if t.area() == 0:
                self.triangles.remove(t)

    def visualize2(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        t_values = np.linspace(0, 2 * np.pi, 100)
        contour_points = self.contour(t_values)
        ax.plot(contour_points[0], contour_points[1], contour_points[2], label='Contour', color='red')

        i = 0
        points = []

        for triangle in self.triangles:
            tri_points = np.array([triangle.d1, triangle.d2, triangle.d3])
            points.append(tri_points)
            tri_collection = Poly3DCollection([tri_points])
            alpha = triangle.layerr/len(self.layers)
            tri_collection.set_facecolor(plt.cm.plasma(alpha))

            ax.add_collection3d(tri_collection)
            i += 1

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        all_points = np.vstack(points)
        ax.set_xlim([min(all_points[:, 0]) - 1, max(all_points[:, 0]) + 1])
        ax.set_ylim([min(all_points[:, 1]) - 1, max(all_points[:, 1]) + 1])
        ax.set_zlim([min(all_points[:, 2]) - 1, max(all_points[:, 2]) + 1])

        ax.legend()
        plt.show()

    def minimize(self):
        for layer in self.layers[1:]:
            for point in layer.points:
                sum0 = point.sum_area()
                ds = []
                for j in range(1):
                    triangle = point.my_triengles[j]
                    distances = [
                        np.linalg.norm(triangle.d1 - triangle.d2),
                        np.linalg.norm(triangle.d1 - triangle.d3),
                        np.linalg.norm(triangle.d2 - triangle.d3)
                    ]
                    ds.extend(distances)

                mini = min(ds) / 10
                dx, dy, dz = mini, mini, mini

                gradients = []
                for delta, axis in zip([dx, dy, dz], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                    direction_vector = np.array([delta * i for i in axis])
                    point.x += direction_vector
                    grad = abs(-point.sum_area() + sum0) / delta if delta != 0 else 0
                    if (point.sum_area() > sum0):
                        point.x -= 2*direction_vector
                        if point.sum_area() > sum0 :
                            grad = 0
                        else:
                            grad = -grad
                        point.x += 2 * direction_vector
                    gradients.append(grad)
                    point.x -= direction_vector

                norm_grad = np.linalg.norm(gradients)
                if norm_grad > 0:
                    point.move = np.array(gradients) / norm_grad * dx

                min_sum = point.sum_area()
                index = 0
                k = 11
                while (index == 0) and (k > -1):
                    point.x += point.move * k
                    current_sum = point.sum_area()
                    if current_sum < min_sum:
                        min_sum = current_sum
                        index = k
                    point.x -= point.move * k
                    if k > 2:
                        k = k - 1
                    elif k > 0.5:
                        k = k - 0.5
                    else:
                        k = k - 0.25

                point.x += point.move * index

        return

    def sum_area(self):
        sum = 0
        for t in self.triangles:
            sum += t.area()
        return(sum)

def contour_func1(t):
    return np.array([6 * np.cos(t), 6 * np.sin(t), 1 * np.sin(4 * t)])
def contour_func2(t):
    return np.array([6 * np.cos(t) + 0.5 * np.cos(4 * t),  6 * np.sin(t) + 0.5 * np.sin(4 * t), 1.5 * np.sin(3 * t)])
def contour_func3(t):
    return np.array([1 * np.sin(t), 1 * np.sin(2 * t), 1 * np.sin(3 * t)])

contour = Contour(contour_func1, N=35, n=201, T_max=2*np.pi)
contour.process()

for _ in range(10):
    contour.minimize()
    print(contour.sum_area())

last_areas = []

contour.visualize2()
