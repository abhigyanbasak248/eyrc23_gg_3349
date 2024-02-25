'''
* Team Id : 3349
* Author List : Abhigyan Basak, Shagnik Guha
* Filename: task_5a.py
* Theme: GeoGuide
* Functions: dijkstra(), print_solution(), print_path(), priority_sort(), main()
* Global Variables: NO_PARENT, SERVER, PORT, event_priority, path, complete_path, start, event_nodes, start_node, end_node, adjacency_matrix
'''


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
SERVER = "192.168.152.194"
PORT = 80

event_priority = {"fire" : 1, "destroyed_buildings" : 2, "human_aid_rehabilitation" : 3, "military_vehicles" : 4, "combat" : 5}
path = []
complete_path = []
start_direction = ['U']

event_nodes = {"A" : 1, "B" : 4, "C" : 10, "D" : 8, "E" : 15}
start_node = 0
end_node = 0

#adjacency matrix: the matrix represents the graph form of the arena. The nodes are considered as vertices and additonal vertices added in front
#                  of the event boxes as the bot has to stop there. The edge weights are given according the distance measured between them on the arena.
#                  Additional two parameters have been added along with weight, row_difference and column_difference. Its the row and column separation
#                  to figure out the direction that the bot needs to go. Lets say the row separation is positive and column separation is 0, that would mean that
#                  the bot needs to move down. If the row separation is 0 and column separation is positive then the bot needs to move right.
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


'''
* Function Name: dijkstra
* Input: adjacency_matrix(adjacency matrix of the graph which represents the arena), 
         start_vertex(the vertex to start the path from),
         dest_vertex(the vertex to reach)
* Output: prints the the start and destination vertex, the distance between them which is the sum of edge weights between them and all
          the intermediary nodes in the path
* Logic: The function initiates by creating essential data structures such as shortest_distances and added to record the shortest 
         distance from the starting vertex to all other vertices and to track which vertices have been processed in the shortest 
         path computation, respectively. It initializes these arrays with default values, setting the distance of the start vertex 
         to itself as zero. Additionally, it establishes the parents array to maintain the parent vertex for each vertex in the shortest 
         path tree, initialized with a special value indicating that the start vertex has no parent. The main loop iterates through the 
         vertices, selecting the nearest unprocessed vertex nearest_vertex and updating distances to its neighbors if a shorter path is 
         found. This process continues until all vertices are processed. By iterating over each vertex in the graph and comparing edge 
         weights, the function efficiently computes the shortest path tree. It then returns the calculated shortest distances and parent 
         vertices, facilitating the reconstruction of shortest paths. This implementation meticulously follows Dijkstra's algorithm, 
         providing an optimal solution for determining shortest paths in weighted graphs.
*
* Example Call: dijkstra(adjacency_matrix, 4, 9)
'''

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
    
    # Calling the print_solution function to print the start and destination nodes, the path cost
    # and the nodes in the path
    # print_solution(start_vertex, shortest_distances, parents, dest_vertex)
    print_path(dest_vertex, parents)



'''
* Function Name: print_solution
* Input: start_vertex(the vertex to start the path from),
         distances(list containing the distance of vertices from start_vertex),
         parents(list containing the parent of each vertex, -1 means no parent),
         dest_vertex(the destination vertex)
* Output: prints the the start and destination vertex, the distance between them which is the sum of the
          weights of the paths and also prints the nodes in the path using the print_path function
* Logic: It initializes vertex_index to the destination vertex and prints the header for the solution. 
         Then, it calls the print_path function to print the path from start_vertex to dest_vertex. 
*
* Example Call: print_solution(0, distances, parents, 7)
'''
# def print_solution(start_vertex, distances, parents, dest_vertex):
#     vertex_index = dest_vertex
#     print(
#         "\n",
#         start_vertex,
#         "->",
#         vertex_index,
#         "\t\t",
#         distances[vertex_index],
#         "\t\t",
#         end="",
#     )
#     # Calling the print_path function to print all the intermediary nodes in the path from start to destination
#     # print_path(vertex_index, parents)
#     path.append(dest_vertex)



'''
* Function Name: print_path
* Input: current_vertex(the current vertex the function is traversing through),
         parents(list containing the parent of each vertex, -1 means no parent)
* Output: prints the shortest path 
* Logic: this function serves as a recursive helper function to print the path from 
         the source to the current vertex by tracing back through the parent vertices. 
         It handles the base case where the current vertex is the source vertex, and 
         recursively prints the parent vertices until reaching the source, effectively 
         printing the shortest path from the start vertex to the destination vertex.
*
* Example Call: print_path(0, parents)
'''
def print_path(current_vertex, parents):
    # Base case : Source node has
    # been processed
    if current_vertex == NO_PARENT:
        return
    print_path(parents[current_vertex], parents)
    # print(current_vertex, end=" ")
    path.append(current_vertex)
    

'''
* Function Name: get_directions
* Input: adjacency_matrix(adjacency matrix of the graph which represents the arena), 
         start_vertex(the vertex to start the path from),
         dest_vertex(the vertex to reach)
* Output: directions list which contains the directions in which the bot needs to move in to reach
          the destination node
* Logic: uses the row and column separation parameter in the adjacency_matrix to set which direction the bot
         need to move in. Lets say the row separation is positive and column separation is 0, that would mean that
         the bot needs to move down. If the row separation is 0 and column separation is positive 
         then the bot needs to move right. The directions are U, D, L, R and S for stay which is for stopping at event
         nodes. This is also represented by the node appearing twice in the path, like [3, 4, 4, 3] means move from 3 to 4
         and stop there and then 4 to 3.
         Last two conditions are for the curved path since the direction of the bit will change when moving along the curved
         path. For curved paths, both row and column separation are non zero.
*
* Example Call: get_directions
'''
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

'''
* Function Name: priority_sort
* Input: event_dic(dictionary containing the class name of event detected for each event, example, {"A" : "fire"})
* Output: returns a sorted dictionary according to the event priority
* Logic: sorts the event dictionary using sorted function according the event_priority dictionary which has priority indicated in terms of 
         integers for each event
*
* Example Call: priority_sort({"A": "combat", "B" : "fire"})
'''
def priority_sort(event_dic):
    sorted_dic = {k: v for k, v in sorted(event_dic.items(), key=lambda item: event_priority.get(item[1], float('inf')))}
    return sorted_dic


'''
* Function Name: main
* Input: None
* Output: None
* Logic: uses file.txt generated from new_task_4a.py using the yolo model. The file contains the prediction in dictionary form
         like {"A" : "fire", "B" : "combat"}. 
*
* Example Call: Called automatically by the Operating System
'''

with open('file.txt', "r") as file:
    event_detected = json.load(file)
new_event = {}
for key, value in zip(event_detected.keys(), event_detected.values()):
    if (value != "blank"):
        new_event[key] = value
# new_event = {"E" : "fire"}

event_detected = priority_sort(new_event)

for event in event_detected.keys():
    dest_node = event_nodes[event]
    dijkstra(adjacency_matrix, start_node, dest_node)
    complete_path += path
    start_node = dest_node
    path = []
dijkstra(adjacency_matrix, start_node, end_node)
complete_path += path

complete_dir = get_directions(complete_path, adjacency_matrix)
complete_dir = start_direction+complete_dir

combined = [complete_path, complete_dir]
print()
# print(combined)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER, PORT))
    json_data = json.dumps(combined)
    s.sendall(json_data.encode())
    data = s.recv(1024)
print(data.decode())