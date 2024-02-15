import json
import socket
import sys
import cv2 as cv
import numpy as np
import torch
from PIL import Image
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

NO_PARENT = -1
# SERVER = "172.20.10.04"
SERVER = "192.168.45.194"
PORT = 80

# event_list = []
# detected_list = {}
# device = "cuda" if torch.cuda.is_available() else "cpu"

combat = "combat"
rehab = "human_aid_rehabilitation"
military_vehicles = "military_vehicles"
fire = "fire"
destroyed_building = "destroyed_buildings"

# arr = [[198,873,55,72],
#        [522,717,55,80],
#        [189,512,55,80],
#        [532,529,60,80],
#        [205,200,55,80]]

# font = cv.FONT_HERSHEY_SIMPLEX
# fontScale = 0.5
# fontColor = (0,255,0)
# thickness = 2
# lineType = 3

priority = {"fire" : 1, "destroyed_buildings" : 2, "human_aid_rehabilitation" : 3, "military_vehicles" : 4, "combat" : 5}
file_path = "file.txt"
# img = cv.imread('/Users/ajitbasak/Desktop/s3.png')
# img = cv.resize(img, (840, 995))

event_nodes = {"A" : 1, "B" : 4, "C" : 10, "D" : 8, "E" : 15}
# event_detected = {"A": "Fire", "E": "Fire"}
start_node = 0
end_node = 0

def dijkstra(adjacency_matrix, start_vertex, dest_vertex):
    n_vertices = len(adjacency_matrix[0])

    # shortest_distances[i] will hold the
    # shortest distance from start_vertex to i
    shortest_distances = [sys.maxsize] * n_vertices

    # added[i] will true if vertex i is
    # included in shortest path tree
    # or shortest distance from start_vertex to
    # i is finalized
    added = [False] * n_vertices

    # Initialize all distances as
    # INFINITE and added[] as false
    for vertex_index in range(n_vertices):
        shortest_distances[vertex_index] = sys.maxsize
        added[vertex_index] = False

    # Distance of source vertex from
    # itself is always 0
    shortest_distances[start_vertex] = 0

    # Parent array to store shortest
    # path tree
    parents = [-1] * n_vertices

    # The starting vertex does not
    # have a parent
    parents[start_vertex] = NO_PARENT

    # Find shortest path for all
    # vertices
    for i in range(1, n_vertices):
        # Pick the minimum distance vertex
        # from the set of vertices not yet
        # processed. nearest_vertex is
        # always equal to start_vertex in
        # first iteration.
        nearest_vertex = -1
        shortest_distance = sys.maxsize
        for vertex_index in range(n_vertices):
            if (
                not added[vertex_index]
                and shortest_distances[vertex_index] < shortest_distance
            ):
                nearest_vertex = vertex_index
                shortest_distance = shortest_distances[vertex_index]

        # Mark the picked vertex as
        # processed
        added[nearest_vertex] = True

        # Update dist value of the
        # adjacent vertices of the
        # picked vertex.
        for vertex_index in range(n_vertices):
            edge_distance = adjacency_matrix[nearest_vertex][vertex_index][0]

            if (
                edge_distance > 0
                and shortest_distance + edge_distance < shortest_distances[vertex_index]
            ):
                parents[vertex_index] = nearest_vertex
                shortest_distances[vertex_index] = shortest_distance + \
                    edge_distance

    print_solution(start_vertex, shortest_distances, parents, dest_vertex)


# A utility function to print
# the constructed distances
# array and shortest paths
def print_solution(start_vertex, distances, parents, dest_vertex):
    vertex_index = dest_vertex
    print(
        "\n",
        start_vertex,
        "->",
        vertex_index,
        "\t\t",
        distances[vertex_index],
        "\t\t",
        end="",
    )
    print_path(vertex_index, parents)


# Function to print shortest path
# from source to current_vertex
# using parents array
path = []


def print_path(current_vertex, parents):
    # Base case : Source node has
    # been processed
    if current_vertex == NO_PARENT:
        return
    print_path(parents[current_vertex], parents)
    print(current_vertex, end=" ")
    path.append(current_vertex)


def get_directions(path, adjacency_matrix):
    directions = []
    start = path[0]
    for i in range(1, len(path)):
        grid = adjacency_matrix[start][path[i]]
        if (start == path[i]):
            directions.append('S')
        elif grid[1] == 0 and grid[2] > 0:
            directions.append("R")
        elif grid[1] == 0 and grid[2] < 0:
            directions.append("L")
        elif grid[2] == 0 and grid[1] > 0:
            directions.append("U")
        elif grid[2] == 0 and grid[1] < 0:
            directions.append("D")
        elif grid[2] > grid[1]:
            if grid[2] > 0:
                directions.append("R")
            else:
                directions.append("L")
        elif grid[1] > grid[2]:
            if grid[1] > 0:
                directions.append("U")
            else:
                directions.append("D")
        start = path[i]
    return directions

def event_priority(event_dic):
    sorted_dic = {k: v for k, v in sorted(event_dic.items(), key=lambda item: priority.get(item[1], float('inf')))}
    return sorted_dic

adjacency_matrix1 = [
    [0, 5, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 8, 0, 15, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 15, 0, 4, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 6, 0, 10, 0, 0, 9, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 10, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 3, 0, 0, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 0, 7, 0, 6, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 4, 0, 6, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 10, 0, 20],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 10, 0, 10, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 10, 0, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 8, 0],
]

adjacency_matrix = [
    [ [0], [ 13, 0, 1 ], [0], [0], [0], [0], [ 19, 1, 0 ], [0], [0], [0], [0], [0], [0], [0], [0], [0] ],
    [ [ 13, 0, -1 ], [0], [ 24, 0, 1 ], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0] ],
    [ [0], [ 24, 0, -1 ], [0], [ 55, 1, 2 ], [0], [ 20, 1, 0 ], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0] ],
    [ [0], [0], [ 55, -1, -2 ], [0], [ 17, 0, -1 ], [0], [0], [0], [0], [0], [0], [ 19, 1, 0 ], [0], [0], [0], [0] ],
    [ [0], [0], [0], [ 17, 0, 1 ], [0], [ 23, 0, -1 ], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0] ],
    [ [0], [0], [ 20, -1, 0 ], [0], [ 23, 0, 1 ], [0], [ 39, 0, -2 ], [0], [0], [ 20, 1, 0 ], [0], [0], [0], [0], [0], [0] ],
    [ [ 19, -1, 0 ], [0], [0], [0], [0], [ 39, 0, 2 ], [0], [ 22, 1, 0 ], [0], [0], [0], [0], [0], [0], [0], [0] ],
    [ [0], [0], [0], [0], [0], [0], [ 22, -1, 0 ], [0], [ 13, 0, 1 ], [0], [0], [0], [0], [0], [ 15, 1, 0 ], [0] ],
    [ [0], [0], [0], [0], [0], [0], [0], [ 13, 0, -1 ], [0], [ 26, 0, 1 ], [0], [0], [0], [0], [0], [0] ],
    [ [0], [0], [0], [0], [0], [ 20, -1, 0 ], [0], [0], [ 26, 0, -1 ], [0], [ 23, 0, 1 ], [0], [0], [ 15, 1, 0 ], [0], [0] ],
    [ [0], [0], [0], [0], [0], [0], [0], [0], [0], [ 23, 0, -1 ], [0], [ 17, 0, 1 ], [0], [0], [0], [0] ],
    [ [0], [0], [0], [ 19, -1, 0 ], [0], [0], [0], [0], [0], [0], [ 17, 0, -1 ], [0], [ 15, 1, 0 ], [0], [0], [0] ],
    [ [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [ 15, -1, 0 ], [0], [ 40, 0, -2 ], [0], [ 78, 2, -3 ] ],
    [ [0], [0], [0], [0], [0], [0], [0], [0], [0], [ 15, -1, 0 ], [0], [0], [ 40, 0, 2 ], [0], [ 39, 0, -2 ], [0] ],
    [ [0], [0], [0], [0], [0], [0], [0], [ 15, -1, 0 ], [0], [0], [0], [0], [0], [ 39, 0, 2 ], [0], [ 26, 2, 1 ] ],
    [ [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [ 78, -2, 3 ], [0], [ 26, -2, -1 ], [0] ],
]


complete_path = []
start = ['U']

with open(file_path, "r") as file:
    event_detected = json.load(file)
new_event = {}
for key, value in zip(event_detected.keys(), event_detected.values()):
    if (value != "blank"):
        new_event[key] = value

# with open(file_path, "w") as file:
#     file.truncate()
event_detected = event_priority(new_event)
print(event_detected)

for event in event_detected.keys():
    dest_node = event_nodes[event]
    dijkstra(adjacency_matrix, start_node, dest_node)
    complete_path += path
    start_node = dest_node
    path = []
dijkstra(adjacency_matrix, start_node, end_node)
complete_path += path
path = []

complete_dir = get_directions(complete_path, adjacency_matrix)
complete_dir = start+complete_dir

combined = [complete_path, complete_dir]
print()
print(combined)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER, PORT))
    json_data = json.dumps(combined)
    s.sendall(json_data.encode())
    data = s.recv(1024)
print(data.decode())