import random
import math
import matplotlib.pyplot as plt

def generate_random_points(num_points, x_range, y_range):
    points = []
    for i in range(num_points):
        x = random.uniform(0,x_range)
        y = random.uniform(0,y_range)

        points.append((x,y))

    return points


def distance(p1, p2):
    return math.sqrt(p1[0]**2 + p1[1]**2) + math.sqrt(p2[0]**2 + p2[1]**2)

def nearest_neighbour(points):
    unvisited = points.copy()
    tour = []
    current_point = unvisited.pop(0)

    tour.append(current_point)

    while unvisited:
        nearest_point = min(unvisited, key=lambda p: distance(current_point, p))
        tour.append(nearest_point)

        unvisited.remove(nearest_point)
        current_point = nearest_point 

    return tour

def plot_tour(tour):
    x = [point[0] for point in tour]
    y = [point[1] for point in tour]

    plt.figure(figsize=(9,8))
    plt.plot(x, y, marker='o')
    plt.title('Nearest Neighbour TSP Tour')
    plt.xlabel('X-axis')
    plt.ylabel("Y-axis")

    plt.grid(True)
    plt.legend(['Tour'])
    plt.show()

#usage

points= generate_random_points(100, random.uniform(0,50), random.uniform(0,50))
tour = nearest_neighbour(points)

plot_tour(tour)