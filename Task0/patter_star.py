def pattern(N):
    for i in range(N, 0, -1):
        for j in range(1, i+1):
            if j%5 == 0:
                print("#", end="")
            else:
                print("*", end="")
        print()
        
def read():
    T = int(input())
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    list = []
    for i in range(T):
        N = int(input())
        if N<0 or N>100:
            exit()
        list.append(N)
    for num in list:
        pattern(num)

if __name__ == "__main__":
    read()