'''
import functools

def generate_AP(a1, d, n):
    series = []
    square = []
    for i in range(1, n+1):
        series.append((a1+(i-1)*d))
    for i in series:
        print(i, end = " ")
    print()
    square = list(map(lambda x: x ** 2, series))
    for i in square:
        print(i, end = " ")
    print()
    sum = functools.reduce(lambda x, y: x+y, square)
    print(sum)
    
def read():
    T = int(input())
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    list = []
    for i in range(T):
        a, d, n = map(int, input().split())
        if 1>a or a>100 or 1>d or d>100 or 1>n or n>100:
            exit()
        list.append([a, d, n])
    for num in list:
        for (a1, d, n) in list:
            generate_AP(a1, d, n)
        print()

if __name__ == "__main__":
    read()
'''

import functools

def generate_AP(a1, d, n):
    AP_series = []
    for i in range(n):
        AP_series.append(a1 + i * d)
    return AP_series

if __name__ == '__main__':
    test_cases = int(input())
    for _ in range(test_cases):
        a1, d, n = map(int, input().split())
        AP_series = generate_AP(a1, d, n)
        print(*AP_series)
        sqr_AP_series = list(map(lambda x: x ** 2, AP_series))
        print(*sqr_AP_series)
        sum_sqr_AP_series = functools.reduce(lambda x, y: x + y, sqr_AP_series)
        print(sum_sqr_AP_series)
