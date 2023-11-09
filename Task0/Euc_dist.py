'''
import math

def cal(x1, y1, x2, y2):
    x_diff = (x1-x2)**2
    y_diff = (y1-y2)**2
    final = math.sqrt((x_diff+y_diff))
    final = f'{final:.2f}'
    print(final)

def read():
    T = int(input())
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    list = []
    for i in range(T):
        x1, y1, x2, y2 = map(int, input().split())
        if not (-100 <= x1 <= 100 and -100 <= y1 <= 100 and -100 <= x2 <= 100 and -100 <= y2 <= 100):
            exit()
        list.append([x1, y1, x2, y2])
    for (x1, y1, x2, y2) in list:
        cal(x1, y1, x2, y2)
        
if __name__ == "__main__":
    read()
'''
import math

def compute_distance(x1, y1, x2, y2):
    x_diff = (x1 - x2) ** 2
    y_diff = (y1 - y2) ** 2
    final = math.sqrt(x_diff + y_diff)
    final = f'{final:.2f}'
    print(final)

def read():
    T = int(input())
    if not (1 <= T <= 25):
        raise ValueError("Invalid input for test cases (1 <= T <= 25)")
    
    points_list = []
    for i in range(T):
        x1, y1, x2, y2 = map(int, input().split())
        if not (-100 <= x1 <= 100 and -100 <= y1 <= 100 and -100 <= x2 <= 100 and -100 <= y2 <= 100):
            raise ValueError("Invalid input for point coordinates (-100 <= x1, y1, x2, y2 <= 100)")
        points_list.append((x1, y1, x2, y2))
    
    for (x1, y1, x2, y2) in points_list:
        compute_distance(x1, y1, x2, y2)

if __name__ == "__main__":
    read()
